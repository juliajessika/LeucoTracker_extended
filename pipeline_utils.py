# pipeline_utils.py
import cv2
import numpy as np
import os
import fnmatch # Для поиска файлов по шаблону
import re
import trackpy as tp
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import TextBox, Button, Slider 
import pims_nd2
import pandas as pd
from pandas import ExcelWriter 
from PIL import Image
import h5py
import imageio
import traceback

DEFAULT_TIME_STEP_SECONDS = 5.0

# --- Функции загрузки ND2 и базовой обработки ---
def get_z_size_nd2(trace_path):
    try:
        with pims_nd2.ND2_Reader(trace_path) as frames:
            z = frames.sizes.get('m') # 'm' is often used for Z in ND2, but could be 'z'
            if z is None: # Check for 'z' if 'm' is not found
                z = frames.sizes.get('z')
            pixel_size = frames[0].metadata.get('mpp', 0.0) # Microns per pixel
            if pixel_size == 0.0: pixel_size = 0.16 # Fallback if not in metadata
            return z, pixel_size
    except Exception as e:
        print(f"Ошибка при чтении Z-размера из {trace_path}: {e}")
        return None, 0.0

def get_nd2_timeframes(trace_path, z_stack_idx, channel_number, file_start_time_offset_sec, time_step_seconds):
    try:
        with pims_nd2.ND2_Reader(trace_path) as frames:
            if z_stack_idx is not None: frames.default_coords['m'] = z_stack_idx
            try: # Attempt to set channel, might not exist or be relevant
                if 'c' in frames.sizes: frames.default_coords['c'] = channel_number
            except Exception: pass 
            t = frames.sizes.get('t')
            if t is None or t == 0: return np.array([])
            times = file_start_time_offset_sec + np.arange(t) * time_step_seconds
            return times
    except Exception as e:
        print(f"Ошибка при расчете временных меток для {trace_path}: {e}")
        return np.array([])

def load_nd2_data(trace_path, z_stack_idx, channel_number,
                    max_chunks_for_splitting=None, current_chunk_idx=None):
    try:
        with pims_nd2.ND2_Reader(trace_path) as frames_reader:
            # Set default coordinates for Z (if provided) and Channel
            if z_stack_idx is not None: 
                frames_reader.default_coords['m'] = z_stack_idx # Assuming 'm' is Z
            if 'c' in frames_reader.sizes: # Only set channel if it exists
                frames_reader.default_coords['c'] = channel_number
            
            x_total, y_total = frames_reader.sizes.get('x'), frames_reader.sizes.get('y')
            t_total = frames_reader.sizes.get('t')

            if t_total is None or t_total == 0:
                print(f"Предупреждение: Файл {os.path.basename(trace_path)} не содержит временных кадров (t_total={t_total}).")
                return None, None
            if x_total is None or y_total is None:
                print(f"Предупреждение: Файл {os.path.basename(trace_path)} не имеет размеров x/y.")
                return None, None

            actual_x, actual_y = x_total, y_total
            current_x_slice, current_y_slice = slice(None), slice(None)

            if z_stack_idx is None and max_chunks_for_splitting is not None and current_chunk_idx is not None and max_chunks_for_splitting > 1:
                num_side = int(np.ceil(np.sqrt(max_chunks_for_splitting)))
                chunk_width = x_total // num_side
                chunk_height = y_total // num_side
                
                row = current_chunk_idx // num_side
                col = current_chunk_idx % num_side
                
                x_start = col * chunk_width
                y_start = row * chunk_height
                
                # Adjust for last chunk in row/col to take remaining pixels
                x_end = (col + 1) * chunk_width if col < num_side - 1 else x_total
                y_end = (row + 1) * chunk_height if row < num_side - 1 else y_total

                current_x_slice = slice(x_start, x_end)
                current_y_slice = slice(y_start, y_end)
                actual_x, actual_y = x_end - x_start, y_end - y_start
            
            img_data_for_chunk = np.zeros((t_total, actual_y, actual_x), dtype=frames_reader.pixel_type)
            
            for i in range(t_total):
                frame_data = np.array(frames_reader[i]) # Should be 2D (Y,X) due to default_coords

                if frame_data.ndim == 3: 
                    print(f"Предупреждение: Кадр {i} из {os.path.basename(trace_path)} имеет 3 измерения ({frame_data.shape}) несмотря на default_coords. Берется срез [0, :, :]. Координаты по умолчанию: {frames_reader.default_coords}")
                    frame_data = frame_data[0, :, :] 
                elif frame_data.ndim != 2:
                    print(f"Ошибка: Кадр {i} из {os.path.basename(trace_path)} имеет неожиданное число измерений ndim={frame_data.ndim}. Форма: {frame_data.shape}. Пропуск кадра.")
                    # Fill with zeros or skip this frame's data
                    img_data_for_chunk[i, :, :] = np.zeros((actual_y, actual_x), dtype=frames_reader.pixel_type)
                    continue
                
                # Ensure the frame_data matches expected chunk dimensions after potential slicing from 3D
                if frame_data.shape[0] != y_total or frame_data.shape[1] != x_total:
                     # This case might happen if the 3D->2D slice was from an unexpected dimension order
                     # Or if pims_nd2 behaves differently than assumed for channel/z selection.
                     # For now, we assume it's (Y,X) matching original y_total, x_total.
                     # If chunking is active, the final slice `frame_data[current_y_slice, current_x_slice]` handles dimensions.
                     pass


                img_data_for_chunk[i, :, :] = frame_data[current_y_slice, current_x_slice]

            times_for_file = np.arange(t_total) * DEFAULT_TIME_STEP_SECONDS 
            return img_data_for_chunk, times_for_file
    except Exception as e:
        print(f"Критическая ошибка при загрузке данных из {trace_path} (z={z_stack_idx}, канал={channel_number}, чанк={current_chunk_idx}/{max_chunks_for_splitting}): {e}")
        traceback.print_exc()
        return None, None

def process_frames_trackpy(image_sequence, diameter=39, link_range=15, memory=3, min_len=4):
    if image_sequence is None or len(image_sequence) == 0: return pd.DataFrame(), {}
    try:
        f = tp.batch(image_sequence.astype(np.float32), diameter=diameter)
        if f.empty: return pd.DataFrame(), {}
        t_linked = tp.link_df(f, search_range=link_range, memory=memory)
        if t_linked.empty: return pd.DataFrame(), {}
        f_filtered = tp.filter_stubs(t_linked, threshold=min_len)
        if f_filtered.empty: return pd.DataFrame(), {}

        dictionary_trajectories = {}
        for particle_id, particle_data in f_filtered.groupby('particle'):
            particle_center_list = []
            for _, row in particle_data.iterrows():
                particle_center_list.append([int(row['frame']), float(row['x']), float(row['y']), float(row['size'])])
            dictionary_trajectories[particle_id] = particle_center_list
        return f_filtered, dictionary_trajectories
    except Exception as e_tp:
        print(f"Ошибка во время трекинга (Trackpy): {e_tp}")
        traceback.print_exc()
        return pd.DataFrame(), {}

class GlobalCellSelectorGUI:
    def __init__(self, all_processed_fields_data, parent_app_logger=None):
        self.all_processed_fields_data = all_processed_fields_data
        self.parent_app_logger = parent_app_logger if parent_app_logger else print 

        if not self.all_processed_fields_data:
            self.log_message("GlobalCellSelectorGUI: Нет данных для отображения.")
            self.fig = None 
            return
        
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception as e_style:
            self.log_message(f"Не удалось применить стиль 'seaborn-v0_8-whitegrid': {e_style}. Используется стиль по умолчанию.")

        self.log_message("Предварительная загрузка последовательностей изображений для всех полей...")
        for idx, field_data_item in enumerate(self.all_processed_fields_data):
            self.log_message(f"  Загрузка изображений для поля {idx}: {field_data_item['original_filename']} - {field_data_item['field_or_chunk_id']}")
            load_p = field_data_item["load_params"]
            img_seq, _ = load_nd2_data(
                load_p["filepath_full"], load_p["z_idx"], load_p["channel"],
                load_p["max_chunks"], load_p["chunk_idx"]
            )
            if img_seq is None:
                self.log_message(f"  ПРЕДУПРЕЖДЕНИЕ: Не удалось предварительно загрузить изображения для поля {idx}. Будет использован плейсхолдер.")
                field_data_item['img_sequence_preloaded'] = None
            else:
                field_data_item['img_sequence_preloaded'] = img_seq
        self.log_message("Предварительная загрузка изображений завершена.")

        self.current_field_idx = 0
        self.img_sequence = None 
        
        self.fig = plt.figure(figsize=(18, 13))
        
        gs_outer = self.fig.add_gridspec(3, 1, height_ratios=[1.2, 10, 1.6], hspace=0.35) 

        gs_top_controls = gs_outer[0, 0].subgridspec(1, 12, wspace=0.5) 

        ax_prev_button = self.fig.add_subplot(gs_top_controls[0, 0:2])
        self.button_prev_field = Button(ax_prev_button, '< Prev Field')
        self.button_prev_field.on_clicked(self.on_prev_field)

        ax_field_label = self.fig.add_subplot(gs_top_controls[0, 2:5]) 
        ax_field_label.axis('off') 
        self.field_display_text_obj = ax_field_label.text(0.5, 0.5, '', ha='center', va='center', 
                                                          transform=ax_field_label.transAxes, fontsize=11)

        ax_next_button = self.fig.add_subplot(gs_top_controls[0, 5:7])
        self.button_next_field = Button(ax_next_button, 'Next Field >')
        self.button_next_field.on_clicked(self.on_next_field)
        
        ax_goto_textbox = self.fig.add_subplot(gs_top_controls[0, 8:12]) 
        self.textbox_goto_field = TextBox(ax_goto_textbox, 'Go to (0-idx): ', initial=str(self.current_field_idx))
        self.textbox_goto_field.on_submit(self.on_goto_field_submit)

        self.ax_main = self.fig.add_subplot(gs_outer[1, 0])
        
        gs_bottom_controls = gs_outer[2, 0].subgridspec(1, 12, wspace=0.3, hspace=0.4)

        ax_slider = self.fig.add_subplot(gs_bottom_controls[0, 0:6]) 

        ax_textbox_traj = self.fig.add_subplot(gs_bottom_controls[0, 6:8]) 
        self.textbox_traj_id = TextBox(ax_textbox_traj, 'TP ID: '); 
        self.textbox_traj_id.on_submit(self.on_traj_id_submit)

        ax_textbox_type = self.fig.add_subplot(gs_bottom_controls[0, 8:10]) 
        self.textbox_cell_type = TextBox(ax_textbox_type, 'Type: '); 
        self.textbox_cell_type.on_submit(self.on_cell_type_submit) 

        ax_button_done = self.fig.add_subplot(gs_bottom_controls[0, 10:12])
        self.button_done = Button(ax_button_done, 'Save & Close'); 
        self.button_done.on_clicked(self.on_done_button)

        self._load_field_data() 

        if self.img_sequence is not None and self.img_sequence.shape[0] > 0:
            self.img_display = self.ax_main.imshow(self.img_sequence[self.current_frame_idx], cmap='gray', alpha=0.9)
        else:
            self.img_display = self.ax_main.imshow(np.zeros((100,100), dtype=np.uint8), cmap='gray', alpha=0.9)
            
        self._plot_trajectories_for_current_field() 
        self._update_title()
        self.ax_main.axis('off')

        num_frames_initial_field = self.current_field_data["num_frames"]
        max_f_idx_initial = num_frames_initial_field - 1 if num_frames_initial_field > 0 else 0
        self.slider_frame = Slider(ax_slider, 'Frame', 0, max_f_idx_initial, 
                                   valinit=self.current_frame_idx, valstep=1, valfmt='%0.0f')
        self.slider_frame.on_changed(self.on_slider_update)
        if max_f_idx_initial < self.slider_frame.valmin: 
            self.slider_frame.ax.set_xlim(self.slider_frame.valmin, self.slider_frame.valmin)
        else:
            self.slider_frame.ax.set_xlim(self.slider_frame.valmin, max_f_idx_initial)
        self.slider_frame.set_val(self.current_frame_idx)

        self.textbox_cell_type.set_val(self.current_cell_type_str)
        self._update_field_display_text() 
        
        self.fig.subplots_adjust(left=0.04, right=0.96, bottom=0.05, top=0.93)

        plt.show(block=False)
        if self.fig and hasattr(self.fig.canvas, 'start_event_loop'):
            self.fig.canvas.start_event_loop(timeout=-1)

    def log_message(self, message):
        if self.parent_app_logger:
            self.parent_app_logger(f"[GlobalGUI] {message}")
        else:
            print(f"[GlobalGUI] {message}")

    def _load_field_data(self):
        self.current_field_data = self.all_processed_fields_data[self.current_field_idx]
        self.img_sequence = self.current_field_data.get('img_sequence_preloaded')

        if self.img_sequence is None:
            self.log_message(f"Ошибка: Предварительно загруженная последовательность img_seq для поля {self.current_field_idx} отсутствует. Используется плейсхолдер.")
            num_expected_frames = self.current_field_data.get("num_frames", 1)
            h, w = (100, 100) 
            for fd_item in self.all_processed_fields_data: # Try to get dims from other fields
                loaded_seq = fd_item.get('img_sequence_preloaded')
                if loaded_seq is not None and loaded_seq.ndim == 3 and loaded_seq.shape[0] > 0:
                    h, w = loaded_seq.shape[1], loaded_seq.shape[2]
                    break
            placeholder_frames = num_expected_frames if num_expected_frames > 0 else 1
            self.img_sequence = np.zeros((placeholder_frames, h, w), dtype=np.uint8)
        
        self.trackpy_trajectories_df = self.current_field_data["tp_df"]
        self.trackpy_trajectories_dict = self.current_field_data["tp_dict"]
        
        existing_ids = [item['gui_id_for_excel_col'] for item in self.current_field_data.get("selected_cells_for_this_field", [])]
        self.current_field_gui_particle_counter = max(existing_ids) if existing_ids else 0
        
        self.current_frame_idx = 0 
        self.current_cell_type_str = 'not specified' 
        self.log_message(f"Загружены данные для поля: {self.current_field_idx} ('{self.current_field_data['original_filename']} - {self.current_field_data['field_or_chunk_id']}')")
        if hasattr(self, 'textbox_cell_type'): # Update textbox if it exists
            self.textbox_cell_type.set_val(self.current_cell_type_str)

    def _plot_trajectories_for_current_field(self):
        self.ax_main.clear()
        if self.img_sequence is not None and \
           self.img_sequence.shape[0] > 0 and \
           0 <= self.current_frame_idx < self.img_sequence.shape[0] and \
           self.current_field_data["num_frames"] > 0:
             self.img_display = self.ax_main.imshow(self.img_sequence[self.current_frame_idx], cmap='gray', alpha=0.9)
        else:
            self.img_display = self.ax_main.imshow(np.zeros((100,100), dtype=np.uint8), cmap='gray', alpha=0.9)
            if self.current_field_data["num_frames"] > 0 :
                self.log_message("Предупреждение: Проблема с последовательностью изображений в _plot_trajectories. Отображен плейсхолдер.")

        if not self.trackpy_trajectories_df.empty:
            tp.plot_traj(self.trackpy_trajectories_df, ax=self.ax_main, label=True, plot_style={'alpha':0.5, 'lw':1}) # lw for line width
        self.ax_main.axis('off')


    def _update_title(self):
        cfd = self.current_field_data
        # User-friendly 1-based indexing for display, frame index also 0-based for consistency with slider
        title = (f"Поле {self.current_field_idx + 1}/{len(self.all_processed_fields_data)}: "
                 f"{cfd['original_filename']} - {cfd['field_or_chunk_id']} ({cfd['interval_name']})\n"
                 f"Кадр: {self.current_frame_idx} (из {cfd['num_frames']-1 if cfd['num_frames']>0 else 0})")
        self.ax_main.set_title(title, fontsize=10)
    
    def _update_field_display_text(self):
        if hasattr(self, 'field_display_text_obj'):
            self.field_display_text_obj.set_text(f"Поле: {self.current_field_idx + 1} / {len(self.all_processed_fields_data)}")
        if hasattr(self, 'textbox_goto_field'):
            self.textbox_goto_field.set_val(str(self.current_field_idx))

    def _change_field_to(self, new_idx):
        if not (0 <= new_idx < len(self.all_processed_fields_data)):
            self.log_message(f"Неверный индекс поля: {new_idx}")
            if hasattr(self, 'textbox_goto_field'):
                 self.textbox_goto_field.set_val(str(self.current_field_idx)) 
            return

        self.current_field_idx = new_idx
        self.log_message(f"Переключение на индекс поля: {self.current_field_idx}")
        
        self._load_field_data() 

        num_frames_new_field = self.current_field_data["num_frames"]
        max_f_idx = num_frames_new_field - 1 if num_frames_new_field > 0 else 0
        
        self.slider_frame.valmax = max_f_idx
        if max_f_idx < self.slider_frame.valmin:
            self.slider_frame.ax.set_xlim(self.slider_frame.valmin, self.slider_frame.valmin)
        else:
            self.slider_frame.ax.set_xlim(self.slider_frame.valmin, max_f_idx)
        self.slider_frame.set_val(0) 
        
        self._plot_trajectories_for_current_field() 
        self._update_title() 
        self._update_field_display_text()
        # self.textbox_cell_type.set_val(self.current_cell_type_str) # Done by _load_field_data
        self.fig.canvas.draw_idle()

    def on_prev_field(self, event):
        if self.current_field_idx > 0:
            self._change_field_to(self.current_field_idx - 1)

    def on_next_field(self, event):
        if self.current_field_idx < len(self.all_processed_fields_data) - 1:
            self._change_field_to(self.current_field_idx + 1)

    def on_goto_field_submit(self, text):
        try:
            new_idx = int(text) 
            self._change_field_to(new_idx)
        except ValueError:
            self.log_message(f"Неверный ввод для 'Перейти к полю': '{text}'. Введите число.")
            self.textbox_goto_field.set_val(str(self.current_field_idx)) 


    def on_slider_update(self, val):
        self.current_frame_idx = int(round(self.slider_frame.val))
        
        num_frames_for_field = self.current_field_data["num_frames"]

        if self.img_sequence is None or self.img_sequence.shape[0] == 0 or num_frames_for_field == 0:
            self.img_display.set_data(np.zeros((100,100), dtype=np.uint8)) 
            self._update_title()
            self.fig.canvas.draw_idle()
            if num_frames_for_field > 0: 
                self.log_message("Нет данных изображения для текущего поля и кадра при обновлении слайдера.")
            return
        
        if self.current_frame_idx >= self.img_sequence.shape[0]:
            self.current_frame_idx = self.img_sequence.shape[0] - 1 if self.img_sequence.shape[0] > 0 else 0
        if self.current_frame_idx < 0: self.current_frame_idx = 0

        self.img_display.set_data(self.img_sequence[self.current_frame_idx])
        self._update_title()
        self.fig.canvas.draw_idle()

    def on_traj_id_submit(self, text):
        try:
            traj_id_trackpy = int(text) 
            if traj_id_trackpy not in self.trackpy_trajectories_dict:
                self.log_message(f"Ошибка: Траектория TrackPy ID {traj_id_trackpy} не найдена."); return
            
            if any(item['particle_id_trackpy'] == traj_id_trackpy for item in self.current_field_data.get("selected_cells_for_this_field", [])):
                self.log_message(f"Траектория TrackPy ID {traj_id_trackpy} уже добавлена для этого поля."); return
            
            self.current_field_gui_particle_counter += 1 
            
            if "selected_cells_for_this_field" not in self.current_field_data:
                 self.current_field_data["selected_cells_for_this_field"] = []

            self.current_field_data["selected_cells_for_this_field"].append({
                'particle_id_trackpy': traj_id_trackpy, 
                'type': self.current_cell_type_str, 
                'gui_id_for_excel_col': self.current_field_gui_particle_counter
            })
            self.log_message(f"Добавлена траектория: ID={traj_id_trackpy}, Тип='{self.current_cell_type_str}', Excel ID={self.current_field_gui_particle_counter} для поля {self.current_field_idx}")
            self.textbox_traj_id.set_val("") 
        except ValueError:
            self.log_message("Ошибка: Введите числовой ID траектории (TrackPy). Поле ввода не очищено.")
        except Exception as e:
            self.log_message(f"Неизвестная ошибка при добавлении траектории: {e}")
            traceback.print_exc()


    def on_cell_type_submit(self, text):
        self.current_cell_type_str = text.strip() if text else "not specified"
        self.log_message(f"Текущий тип клетки: '{self.current_cell_type_str}'")
        self.textbox_cell_type.set_val(self.current_cell_type_str) 

    def on_done_button(self, event):
        self.log_message("Завершение выбора клеток (нажата кнопка 'Save & Close').")
        if self.fig:
            plt.close(self.fig)
            if hasattr(self.fig.canvas, 'stop_event_loop'):
                 try:
                    # Check if manager exists and has key_press_handler_id, a loose proxy for active loop
                    if self.fig.canvas.manager and hasattr(self.fig.canvas.manager, 'key_press_handler_id'):
                         self.fig.canvas.stop_event_loop()
                 except AttributeError: 
                    pass 
                 except Exception as e: 
                    self.log_message(f"Небольшая ошибка при остановке цикла событий: {e}")

# ... (rest of pipeline_utils.py: save_field_data_to_excel, sorting functions, Excel processing, H5/Cellpose) ...
# The functions from save_field_data_to_excel downwards remain unchanged by this request.
# Make sure they are all present in your file.
def save_field_data_to_excel(trackpy_trajectories_dict,
                             original_filename, field_or_chunk_id, interval_id,
                             current_file_global_start_time_sec, output_path_base,
                             selected_cell_info_list_with_gui_id, 
                             time_step_seconds
                            ):
    if not selected_cell_info_list_with_gui_id:
        print(f"[SaveExcel] Нет выбранных клеток для {original_filename} - {field_or_chunk_id}.")
        return

    coords_data_dict = {}
    mean_values_row_data = {} 
    max_len_trajectory = 0

    for sel_cell in selected_cell_info_list_with_gui_id:
        tp_id, cell_type, gui_id = sel_cell['particle_id_trackpy'], sel_cell['type'], sel_cell['gui_id_for_excel_col']
        trajectory_points = trackpy_trajectories_dict.get(tp_id)
        if not trajectory_points:
            print(f"[SaveExcel] Предупреждение: не найдены точки для tp_id {tp_id} в {original_filename}-{field_or_chunk_id}")
            continue

        frames_local = [p[0] for p in trajectory_points]
        times_global_sec = [(current_file_global_start_time_sec + frame_local * time_step_seconds) for frame_local in frames_local]
        coords_x, coords_y, sizes = [p[1] for p in trajectory_points], [p[2] for p in trajectory_points], [p[3] for p in trajectory_points]
        max_len_trajectory = max(max_len_trajectory, len(frames_local))

        coords_data_dict[f'Coords X_{gui_id}'] = coords_x
        coords_data_dict[f'Coords Y_{gui_id}'] = coords_y
        coords_data_dict[f'T №{gui_id}'] = times_global_sec
        coords_data_dict[f'Type №{gui_id}'] = [cell_type] * len(frames_local)
        coords_data_dict[f'Frame_local_{gui_id}'] = frames_local
        coords_data_dict[f'Size_{gui_id}'] = sizes
        coords_data_dict[f'Trackpy_ID_{gui_id}'] = [tp_id] * len(frames_local)

        mean_velocity_pix_per_sec, trajectory_length_pix, trajectory_duration_sec = 0, 0, 0
        if len(coords_x) > 1:
            trajectory_length_pix = np.sum(np.sqrt(np.diff(coords_x)**2 + np.diff(coords_y)**2))
            trajectory_duration_sec = times_global_sec[-1] - times_global_sec[0] if len(times_global_sec) > 1 else 0
            if trajectory_duration_sec > 1e-6: mean_velocity_pix_per_sec = trajectory_length_pix / trajectory_duration_sec
        
        mean_values_row_data[f'Type №{gui_id}'] = cell_type
        mean_values_row_data[f'Veloc{gui_id}'] = mean_velocity_pix_per_sec

    for col_name, col_values in coords_data_dict.items():
        if len(col_values) < max_len_trajectory:
            coords_data_dict[col_name].extend([np.nan] * (max_len_trajectory - len(col_values)))
    df_coords = pd.DataFrame(coords_data_dict)
    df_mean_values = pd.DataFrame([mean_values_row_data]) 

    safe_orig_fname = "".join(c if c.isalnum() else "_" for c in os.path.splitext(original_filename)[0])
    safe_field_id = "".join(c if c.isalnum() else "" for c in str(field_or_chunk_id)) 
    excel_fname = f"{safe_orig_fname}_field{safe_field_id}.xlsx"
    output_dir_interval = os.path.join(output_path_base, f"interval_{interval_id}_results")
    os.makedirs(output_dir_interval, exist_ok=True)
    excel_filepath = os.path.join(output_dir_interval, excel_fname)

    try:
        with ExcelWriter(excel_filepath, engine='openpyxl') as writer:
            df_coords.to_excel(writer, sheet_name='Coords', index=False)
            df_mean_values.to_excel(writer, sheet_name='Mean values', index=False)
        print(f"[SaveExcel] Сохранены данные: {excel_filepath}")
    except Exception as e:
        print(f"[SaveExcel] Ошибка сохранения Excel {excel_filepath}: {e}")
        traceback.print_exc()

def atoi(text): return int(text) if text.isdigit() else text
def natural_keys(text): return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_excel_field_files(path, file_base_name_pattern):
    listOfFiles = os.listdir(path)
    filenames = []
    actual_pattern = file_base_name_pattern + "_field*.xlsx"
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry.lower(), actual_pattern.lower()) and entry.lower().endswith(('.xls', '.xlsx')):
            filenames.append(entry)
    
    def sort_key_field_files(fname):
        match = re.search(r'_field([a-zA-Z0-9]+)\.(xlsx?)$', fname, re.IGNORECASE) 
        if match: return natural_keys(match.group(1))
        return natural_keys(fname)
    filenames.sort(key=sort_key_field_files)
    return filenames

def get_nd2_files_sorted(path):
    listOfFiles = os.listdir(path)
    filenames = [f for f in listOfFiles if f.lower().endswith('.nd2')]
    filenames.sort() 
    return filenames

def all_names_from_excel_v2(folder_with_excel_files, base_filename_for_fields):
    leuko_names_set = set()
    entry_list = get_excel_field_files(folder_with_excel_files, base_filename_for_fields)

    if not entry_list:
        return ['all_names']

    for file_name_short in entry_list:
        file_path_full = os.path.join(folder_with_excel_files, file_name_short)
        try:
            data_mean_vals = pd.read_excel(file_path_full, sheet_name='Mean values')
            if data_mean_vals.empty: continue
            headers = list(data_mean_vals.columns)
            for header_col_name in headers:
                if header_col_name.startswith('Type №'): 
                    cell_type_value = data_mean_vals[header_col_name].iloc[0] 
                    if pd.notna(cell_type_value): leuko_names_set.add(str(cell_type_value))
        except Exception as e: print(f"Ошибка [all_names_v2] при чтении '{file_path_full}': {e}")
    result_list = sorted([name for name in leuko_names_set]) + ['all_names']
    return result_list

def find_related_columns_excel_v2(headers, base_col_name_with_id, related_patterns_map_with_id_placeholder):
    match_id = re.search(r'_(\d+)$', base_col_name_with_id) 
    if not match_id: return None
    
    cell_id_num_str = match_id.group(1)
    related_cols = {}
    for pattern_key, pattern_template in related_patterns_map_with_id_placeholder.items():
        target_col_name = pattern_template.format(id=cell_id_num_str)
        if target_col_name in headers:
            related_cols[pattern_key] = target_col_name
    return related_cols

def get_trajectory_lengths_v2(folder_with_excel_files, base_filename_for_fields, target_cell_type_name):
    traj_lengths_pixels = []
    entry_list = get_excel_field_files(folder_with_excel_files, base_filename_for_fields)
    if not entry_list: return []

    coord_patterns = {'Y': 'Coords Y_{id}', 'Type': 'Type №{id}'} 

    for file_name_short in entry_list:
        file_path_full = os.path.join(folder_with_excel_files, file_name_short)
        try:
            data_coords = pd.read_excel(file_path_full, sheet_name='Coords', header=0)
            if data_coords.empty: continue
            all_headers = list(data_coords.columns)
            
            processed_gui_ids = set() 

            for x_col_name in all_headers:
                if x_col_name.startswith('Coords X_'):
                    gui_id_match = re.search(r'_(\d+)$', x_col_name)
                    if not gui_id_match: continue
                    current_gui_id_str = gui_id_match.group(1)
                    if current_gui_id_str in processed_gui_ids: continue
                    
                    related = find_related_columns_excel_v2(all_headers, x_col_name, coord_patterns)
                    y_col_name = related.get('Y')
                    type_col_name_coords = related.get('Type') 

                    if y_col_name and type_col_name_coords:
                        cell_type_series = data_coords[type_col_name_coords].dropna()
                        if cell_type_series.empty: continue
                        actual_cell_type = str(cell_type_series.iloc[0])

                        if target_cell_type_name == 'all_names' or target_cell_type_name == actual_cell_type:
                            x_coords = data_coords[x_col_name].dropna().tolist()
                            y_coords = data_coords[y_col_name].dropna().tolist()
                            if len(x_coords) != len(y_coords) or len(x_coords) < 2: continue
                            current_traj_len = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
                            traj_lengths_pixels.append(current_traj_len)
                            processed_gui_ids.add(current_gui_id_str)
        except Exception as e: print(f"Ошибка [get_traj_lengths_v2] в '{file_path_full}': {e}")
    return traj_lengths_pixels

def get_trajectory_durations_v2(folder_with_excel_files, base_filename_for_fields, target_cell_type_name):
    traj_durations_sec = []
    entry_list = get_excel_field_files(folder_with_excel_files, base_filename_for_fields)
    if not entry_list: return []

    for file_name_short in entry_list:
        file_path_full = os.path.join(folder_with_excel_files, file_name_short)
        try:
            data_coords = pd.read_excel(file_path_full, sheet_name='Coords', header=0)
            if data_coords.empty: continue
            all_headers = list(data_coords.columns)
            processed_gui_ids = set()

            for header_name in all_headers: 
                if header_name.startswith('T №'): 
                    gui_id_match = re.match(r'T №(?:_)?(\d+)', header_name)
                    if not gui_id_match: continue
                    current_gui_id_str = gui_id_match.group(1)

                    if current_gui_id_str in processed_gui_ids: continue
                    
                    time_col_name = header_name 
                    type_col_name_coords = f'Type №{current_gui_id_str}' 
                    
                    if type_col_name_coords not in all_headers: continue

                    cell_type_series = data_coords[type_col_name_coords].dropna()
                    if cell_type_series.empty: continue
                    actual_cell_type = str(cell_type_series.iloc[0])

                    if target_cell_type_name == 'all_names' or target_cell_type_name == actual_cell_type:
                        time_values_sec = data_coords[time_col_name].dropna().tolist()
                        if len(time_values_sec) >= 2:
                            traj_durations_sec.append(np.max(time_values_sec) - np.min(time_values_sec))
                        elif len(time_values_sec) == 1: traj_durations_sec.append(0.0)
                        processed_gui_ids.add(current_gui_id_str)
        except Exception as e: print(f"Ошибка [get_traj_durations_v2] в '{file_path_full}': {e}")
    return traj_durations_sec


def aggregate_and_save_final_data_v2(list_of_folders_with_excel_field_files,
                                   base_filename_prefix_for_fields, 
                                   output_excel_basename_for_type,  
                                   output_folder_for_final_excel,
                                   velocity_threshold_fast_pix_per_sec=0.045):
    print(f"\nАнализ для префикса: {base_filename_prefix_for_fields}")
    
    all_cell_types_found = set()
    for folder_path in list_of_folders_with_excel_field_files:
        types_in_folder = all_names_from_excel_v2(folder_path, base_filename_prefix_for_fields)
        for t_name in types_in_folder:
            if t_name != 'all_names': all_cell_types_found.add(t_name)
    
    final_types_to_process = sorted(list(all_cell_types_found)) + ['all_names']
    if not any(t != 'all_names' for t in final_types_to_process):
        print(f"  Не найдено типов клеток для '{base_filename_prefix_for_fields}'.")

    print(f"  Типы для анализа: {final_types_to_process}")

    per_cell_data_by_type = {name: [] for name in final_types_to_process}

    for folder_path in list_of_folders_with_excel_field_files:
        entry_list = get_excel_field_files(folder_path, base_filename_prefix_for_fields)
        for file_name_short in entry_list:
            file_path_full = os.path.join(folder_path, file_name_short)
            try:
                df_coords = pd.read_excel(file_path_full, sheet_name='Coords', header=0)
                df_mean_vals = pd.read_excel(file_path_full, sheet_name='Mean values', header=0)
                if df_coords.empty or df_mean_vals.empty : continue

                headers_coords = list(df_coords.columns)
                headers_mean = list(df_mean_vals.columns)

                found_gui_ids_in_file = set()
                for h_coord in headers_coords:
                    m = re.search(r'Coords X_(\d+)', h_coord) 
                    if m: found_gui_ids_in_file.add(m.group(1))
                
                for gui_id_str in found_gui_ids_in_file:
                    type_col_mean = f'Type №{gui_id_str}'
                    vel_col_mean = f'Veloc{gui_id_str}' 

                    if type_col_mean not in headers_mean or vel_col_mean not in headers_mean:
                        continue
                    
                    cell_type_actual = str(df_mean_vals[type_col_mean].iloc[0])
                    velocity_val_mean = df_mean_vals[vel_col_mean].iloc[0]
                    if pd.isna(velocity_val_mean): continue 
                    velocity_actual = float(velocity_val_mean)


                    x_col_coords = f'Coords X_{gui_id_str}'
                    y_col_coords = f'Coords Y_{gui_id_str}'
                    len_L = 0
                    if x_col_coords in headers_coords and y_col_coords in headers_coords:
                        x_vals = df_coords[x_col_coords].dropna().tolist()
                        y_vals = df_coords[y_col_coords].dropna().tolist()
                        if len(x_vals) >= 2 and len(x_vals) == len(y_vals):
                            len_L = np.sum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
                    
                    time_col_coords = f'T №{gui_id_str}' 
                    dur_T = 0
                    if time_col_coords in headers_coords:
                        time_vals = df_coords[time_col_coords].dropna().tolist()
                        if len(time_vals) >= 2:
                            dur_T = np.max(time_vals) - np.min(time_vals)
                    
                    cell_record = {'V': velocity_actual, 'L': len_L, 'T': dur_T, 
                                   'source_file': file_name_short, 'gui_id': gui_id_str}
                    
                    if cell_type_actual in per_cell_data_by_type:
                        per_cell_data_by_type[cell_type_actual].append(cell_record)
                    per_cell_data_by_type['all_names'].append(cell_record)

            except Exception as e: print(f"Ошибка при обработке файла {file_path_full} в aggregate_v2: {e}")

    os.makedirs(output_folder_for_final_excel, exist_ok=True)
    for type_name_to_save in final_types_to_process:
        cell_records_for_this_type = per_cell_data_by_type[type_name_to_save]
        if not cell_records_for_this_type and type_name_to_save != 'all_names':
            continue 

        velocities_all = [rec['V'] for rec in cell_records_for_this_type]
        lengths_all = [rec['L'] for rec in cell_records_for_this_type]
        durations_all = [rec['T'] for rec in cell_records_for_this_type]

        velocities_fast, lengths_fast, durations_fast = [], [], []
        for rec in cell_records_for_this_type:
            if rec['V'] > velocity_threshold_fast_pix_per_sec:
                velocities_fast.append(rec['V'])
                lengths_fast.append(rec['L'])
                durations_fast.append(rec['T'])
        
        df_out_dict = {
            'Velocities': velocities_all, 'Trajectories': lengths_all, 'Trajectories_Time': durations_all,
            'Velocities > thr': velocities_fast, 'Trajectories > thr': lengths_fast, 'Trajectories_TimeFast': durations_fast
        }
        df_final = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in df_out_dict.items() ]))

        
        safe_type_name = "".join(c if c.isalnum() else "_" for c in type_name_to_save)
        final_excel_filename = f"{output_excel_basename_for_type}_{safe_type_name}_fin.xlsx"
        final_excel_filepath = os.path.join(output_folder_for_final_excel, final_excel_filename)
        try:
            df_final.to_excel(final_excel_filepath, index=False)
            print(f"  Сохранен: {final_excel_filepath} (клеток: {len(velocities_all)})")
        except Exception as e: print(f"  Ошибка сохранения {final_excel_filepath}: {e}")


def save_frame_as_h5_tiff(absolute_target_time_sec, experiment_folder_path,
                           ordered_video_files_for_experiment,
                           time_step_seconds, channel_number, output_prefix,
                           pixel_size_microns=0.64, output_folder_override=None):
    print(f"\nПоиск кадра для времени ~{absolute_target_time_sec:.2f} сек...")
    current_cumulative_time = 0.0; found_file_info = None
    for filename in ordered_video_files_for_experiment:
        filepath = os.path.join(experiment_folder_path, filename)
        try:
            with pims_nd2.ND2_Reader(filepath) as frames: num_frames_in_file = frames.sizes.get('t', 0)
            if num_frames_in_file == 0: continue
            file_duration_sec = num_frames_in_file * time_step_seconds
            if absolute_target_time_sec >= current_cumulative_time - 1e-6 and absolute_target_time_sec < current_cumulative_time + file_duration_sec + 1e-6 : 
                time_into_this_file = absolute_target_time_sec - current_cumulative_time
                target_frame_index_in_file = int(round(time_into_this_file / time_step_seconds))
                if target_frame_index_in_file < 0: target_frame_index_in_file = 0
                if target_frame_index_in_file >= num_frames_in_file: target_frame_index_in_file = num_frames_in_file - 1
                calculated_frame_actual_time = current_cumulative_time + target_frame_index_in_file * time_step_seconds
                found_file_info = {"filepath": filepath, "frame_index": target_frame_index_in_file, "calculated_time": calculated_frame_actual_time}
                break
            current_cumulative_time += file_duration_sec
        except Exception as e_read: print(f"  Предупреждение: Не удалось прочитать {filename}: {e_read}."); continue
    if not found_file_info: print(f"Целевое время {absolute_target_time_sec:.2f}с не найдено."); return None

    save_to_folder = output_folder_override if output_folder_override else experiment_folder_path
    os.makedirs(save_to_folder, exist_ok=True)
    try:
        with pims_nd2.ND2_Reader(found_file_info["filepath"]) as frames_reader:
            if 'c' in frames_reader.sizes: frames_reader.default_coords['c'] = channel_number
            # Assuming 'm' is the Z-axis. If files can have multiple Z-stacks and we want the first one:
            if 'm' in frames_reader.sizes: frames_reader.default_coords['m'] = 0 
            elif 'z' in frames_reader.sizes: frames_reader.default_coords['z'] = 0

            frame_array = np.array(frames_reader[found_file_info["frame_index"]])
            if frame_array.ndim == 3: # If still 3D (e.g. color channel not fully selected by pims)
                print(f"Warning: Frame for H5/TIFF is 3D ({frame_array.shape}). Taking first channel/slice.")
                frame_array = frame_array[0] # take first channel or z-slice

        h5_filename = os.path.join(save_to_folder, f"{output_prefix}_clot_{int(round(absolute_target_time_sec))}s.h5")
        tiff_filename = os.path.join(save_to_folder, f"{output_prefix}_clot_{int(round(absolute_target_time_sec))}s.tiff")
        with h5py.File(h5_filename, 'w') as hf: hf.create_dataset('image', data=frame_array)

        if np.issubdtype(frame_array.dtype, np.integer):
            max_v, min_v = np.max(frame_array), np.min(frame_array)
            if max_v == min_v : ar_scaled = frame_array.astype(np.uint16) 
            else: ar_scaled = ((frame_array.astype(np.float64)-min_v)*(65535/(max_v-min_v))).astype(np.uint16)
        elif np.issubdtype(frame_array.dtype, np.floating):
            max_v, min_v = np.max(frame_array), np.min(frame_array)
            if max_v == min_v : ar_scaled = np.zeros_like(frame_array, dtype=np.uint16)
            else: ar_scaled = ((frame_array-min_v)/(max_v-min_v)*65535).astype(np.uint16)
        else: ar_scaled = frame_array.astype(np.uint16) # Ensure uint16 for tiff
        imageio.imwrite(tiff_filename, ar_scaled)
        print(f"Сохранен кадр для ~{absolute_target_time_sec}с ({found_file_info['calculated_time']:.2f}с):")
        print(f"  H5: {h5_filename}\n  TIFF: {tiff_filename}"); return h5_filename
    except Exception as e: print(f"Ошибка извлечения/сохранения кадра для H5/TIFF: {e}"); traceback.print_exc(); return None


def run_cellpose_segmentation(h5_filepath, model_type='cyto', diameter=None, output_folder=None):
    if not os.path.exists(h5_filepath): print(f"Ошибка: H5 файл не найден: {h5_filepath}"); return None
    try:
        from cellpose import models, io, plot 
        with h5py.File(h5_filepath, 'r') as hf:
            img = hf['image'][:] if 'image' in hf else (hf['clot'][:] if 'clot' in hf else (hf['data'][:] if 'data' in hf else None))
        if img is None: print(f"Ошибка: нет датасета ('image', 'clot', or 'data') в {h5_filepath}"); return None
        if img.ndim > 2 and img.shape[0] == 1: img = img[0] # If (1,Y,X), reduce to (Y,X)
        elif img.ndim > 2 : 
            print(f"Предупреждение: Изображение для Cellpose имеет {img.ndim} измерений ({img.shape}). Используется срез [0,:,:].")
            img = img[0] 
        
        model = models.CellposeModel(gpu=False, model_type=model_type) 
        masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=[0,0], progress=None) 

        if output_folder is None: output_folder = os.path.dirname(h5_filepath)
        base_name = os.path.splitext(os.path.basename(h5_filepath))[0]
        
        mask_path = os.path.join(output_folder, f"{base_name}_cellpose_mask.png")
        io.imsave(mask_path, masks.astype(np.uint16))
        print(f"Сохранена маска Cellpose: {mask_path}")

        img_display = img.copy()
        if img_display.dtype != np.uint8:
             min_val, max_val = np.min(img_display), np.max(img_display)
             if max_val > min_val: 
                 img_display = (img_display - min_val) / (max_val - min_val) * 255
             else: # All pixels are the same value
                 img_display = np.zeros_like(img_display) 
             img_display = img_display.astype(np.uint8)
        
        img_rgb_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB) if img_display.ndim == 2 else img_display
        if img_rgb_display.ndim == 3 and img_rgb_display.shape[2] == 1: # If grayscale but 3D
             img_rgb_display = cv2.cvtColor(img_rgb_display, cv2.COLOR_GRAY2RGB)

        overlay_img = plot.mask_overlay(img_rgb_display, masks)
        overlay_path = os.path.join(output_folder, f"{base_name}_cellpose_overlay.png")
        imageio.imwrite(overlay_path, overlay_img)
        print(f"Сохранен оверлей: {overlay_path}")
        return mask_path, overlay_path
    except ImportError: print("Ошибка: cellpose не найден. Убедитесь, что он установлен."); return None
    except Exception as e: print(f"Ошибка Cellpose для {h5_filepath}: {e}"); traceback.print_exc(); return None