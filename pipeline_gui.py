# pipeline_gui.py
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import os
import threading
import numpy as np
import pandas as pd
import queue # Для взаимодействия потоков с GUI
import traceback # Для полного трейсбека
import matplotlib.pyplot as plt
# Импортируем наши утилиты
import pipeline_utils as pu

# Убедимся, что matplotlib использует TkAgg бэкенд для совместимости с Tkinter
import matplotlib
matplotlib.use('TkAgg') # Делать это ДО импорта pyplot


class PipelineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Пайплайн обработки изображений клеток")
        self.root.geometry("850x750")

        self.experiment_path = tk.StringVar()
        self.channel_var = tk.IntVar(value=1)
        self.time_step_var = tk.DoubleVar(value=pu.DEFAULT_TIME_STEP_SECONDS)
        self.patient_name_var = tk.StringVar(value="PatientX")

        self.interval_data = {
            "0-10 min": {"label": "Интервал 0-10 мин (0-600 сек)", "files": [], "listbox": None, "offset_sec": 0, "id_for_folder": 0},
            "10-20 min": {"label": "Интервал 10-20 мин (600-1200 сек)", "files": [], "listbox": None, "offset_sec": 600, "id_for_folder": 1},
            "20-30 min": {"label": "Интервал 20-30 мин (1200-1800 сек)", "files": [], "listbox": None, "offset_sec": 1200, "id_for_folder": 2}
        }
        self.all_nd2_files_in_folder = []

        self.tp_diameter_var = tk.IntVar(value=49)
        self.tp_link_range_var = tk.IntVar(value=27)
        self.tp_memory_var = tk.IntVar(value=3)
        self.tp_min_len_var = tk.IntVar(value=7)

        self.h5_times_str_var = tk.StringVar(value="300, 600, 1500")
        self.cp_model_type_var = tk.StringVar(value="cyto")
        self.cp_diameter_var = tk.IntVar(value=0)

        self.input_queue = queue.Queue()
        self.log_queue = queue.Queue()

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing_main_window)
        self.processing_thread_active = False
        self.active_thread = None

        self.root.after(100, self.process_log_queue)


    def on_closing_main_window(self):
        if self.processing_thread_active and self.active_thread and self.active_thread.is_alive():
            if messagebox.askokcancel("Выход", "Идет обработка. Действительно выйти? Это может повредить данные.", parent=self.root):
                self.processing_thread_active = False 
                self.root.destroy()
            else:
                return
        else:
            self.root.destroy()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        path_frame = ttk.LabelFrame(main_frame, text="1. Выбор папки эксперимента")
        path_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Entry(path_frame, textvariable=self.experiment_path, width=60).pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        ttk.Button(path_frame, text="Обзор...", command=self.browse_folder).pack(side=tk.LEFT, padx=5, pady=5)

        settings_frame = ttk.LabelFrame(main_frame, text="2. Общие настройки")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        sf_inner = ttk.Frame(settings_frame); sf_inner.pack(pady=5)
        ttk.Label(sf_inner, text="Канал (0,1,2...):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(sf_inner, textvariable=self.channel_var, width=5).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(sf_inner, text="Шаг времени (сек):").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(sf_inner, textvariable=self.time_step_var, width=5).grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)
        ttk.Label(sf_inner, text="Имя пациента/эксп.:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(sf_inner, textvariable=self.patient_name_var, width=30).grid(row=1, column=1, columnspan=3, padx=5, pady=2, sticky=tk.W)
        
        intervals_outer_frame = ttk.LabelFrame(main_frame, text="3. Распределение .nd2 файлов по интервалам")
        intervals_outer_frame.pack(fill=tk.X, padx=5, pady=5)
        available_files_frame = ttk.Frame(intervals_outer_frame); available_files_frame.pack(fill=tk.X, pady=5)
        self.available_files_listbox = tk.Listbox(available_files_frame, selectmode=tk.EXTENDED, height=6, exportselection=False)
        self.available_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        available_files_scrollbar = ttk.Scrollbar(available_files_frame, orient=tk.VERTICAL, command=self.available_files_listbox.yview)
        available_files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y); self.available_files_listbox.config(yscrollcommand=available_files_scrollbar.set)
        ttk.Button(intervals_outer_frame, text="Обновить список файлов из папки", command=self.refresh_nd2_file_list).pack(pady=5)
        intervals_selection_frame = ttk.Frame(intervals_outer_frame); intervals_selection_frame.pack(fill=tk.X, expand=True, pady=5)
        col_idx = 0
        for interval_key, data in self.interval_data.items():
            frame = ttk.Frame(intervals_selection_frame); frame.grid(row=0, column=col_idx, padx=10, pady=5, sticky="nsew")
            intervals_selection_frame.grid_columnconfigure(col_idx, weight=1)
            ttk.Label(frame, text=data["label"]).pack(pady=2, anchor=tk.CENTER)
            listbox_frame = ttk.Frame(frame); listbox_frame.pack(pady=2, fill=tk.BOTH, expand=True)
            data["listbox"] = tk.Listbox(listbox_frame, selectmode=tk.SINGLE, height=5, exportselection=False)
            data["listbox"].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            lb_scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=data["listbox"].yview)
            lb_scrollbar.pack(side=tk.RIGHT, fill=tk.Y); data["listbox"].config(yscrollcommand=lb_scrollbar.set)
            btn_frame = ttk.Frame(frame); btn_frame.pack(fill=tk.X, pady=2)
            ttk.Button(btn_frame, text="Добавить", command=lambda k=interval_key: self.add_selected_to_interval(k)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            ttk.Button(btn_frame, text="Удалить", command=lambda k=interval_key: self.remove_selected_from_interval(k)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            col_idx += 1
        
        processing_main_frame = ttk.LabelFrame(main_frame, text="4. Обработка и анализ")
        processing_main_frame.pack(fill=tk.X, padx=5, pady=5)
        tp_params_frame = ttk.Frame(processing_main_frame); tp_params_frame.pack(fill=tk.X, pady=3)
        ttk.Label(tp_params_frame, text="TrackPy ->").pack(side=tk.LEFT, padx=2)
        ttk.Label(tp_params_frame, text="Диаметр:").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(tp_params_frame, textvariable=self.tp_diameter_var, width=3).pack(side=tk.LEFT)
        ttk.Label(tp_params_frame, text="Поиск(px):").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(tp_params_frame, textvariable=self.tp_link_range_var, width=3).pack(side=tk.LEFT)
        ttk.Label(tp_params_frame, text="Память:").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(tp_params_frame, textvariable=self.tp_memory_var, width=3).pack(side=tk.LEFT)
        ttk.Label(tp_params_frame, text="Мин.длина:").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(tp_params_frame, textvariable=self.tp_min_len_var, width=3).pack(side=tk.LEFT)
        ttk.Button(processing_main_frame, text="А. Запустить обработку интервалов (Трекинг и GUI выбора клеток)", command=self.run_interval_processing_thread).pack(fill=tk.X, pady=3)
        ttk.Button(processing_main_frame, text="Б. Собрать итоговые 'finals' Excel файлы", command=self.run_finals_assembly_thread).pack(fill=tk.X, pady=3)
        h5_cp_frame = ttk.Frame(processing_main_frame); h5_cp_frame.pack(fill=tk.X, pady=3)
        ttk.Label(h5_cp_frame, text="H5/Cellpose ->").pack(side=tk.LEFT, padx=2)
        ttk.Label(h5_cp_frame, text="Времена H5(сек):").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(h5_cp_frame, textvariable=self.h5_times_str_var, width=15).pack(side=tk.LEFT)
        ttk.Label(h5_cp_frame, text="CP Модель:").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(h5_cp_frame, textvariable=self.cp_model_type_var, width=8).pack(side=tk.LEFT)
        ttk.Label(h5_cp_frame, text="CP Диаметр(0=авто):").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(h5_cp_frame, textvariable=self.cp_diameter_var, width=3).pack(side=tk.LEFT)
        ttk.Button(processing_main_frame, text="В. Нарезать H5 тромбов и запустить Cellpose", command=self.run_h5_slicing_and_cellpose_thread).pack(fill=tk.X, pady=3)

        log_frame = ttk.LabelFrame(main_frame, text="Лог выполнения")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text_widget = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED, relief=tk.SUNKEN, borderwidth=1)
        log_scrollbar = ttk.Scrollbar(log_frame, command=self.log_text_widget.yview)
        self.log_text_widget.configure(yscrollcommand=log_scrollbar.set)
        self.log_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def safe_log(self, message):
        self.log_queue.put(str(message))

    def process_log_queue(self):
        try:
            while True: 
                message = self.log_queue.get_nowait()
                if self.root and self.root.winfo_exists() and self.log_text_widget:
                    self.log_text_widget.configure(state=tk.NORMAL)
                    self.log_text_widget.insert(tk.END, message + "\n")
                    self.log_text_widget.configure(state=tk.DISABLED)
                    self.log_text_widget.see(tk.END)
        except queue.Empty:
            pass 
        except Exception as e:
            print(f"Ошибка в process_log_queue: {e}") 
        finally:
            if self.root and self.root.winfo_exists(): 
                self.root.after(100, self.process_log_queue)

    def browse_folder(self):
        path = filedialog.askdirectory(parent=self.root)
        if path: self.experiment_path.set(path); self.refresh_nd2_file_list()

    def refresh_nd2_file_list(self):
        exp_path = self.experiment_path.get()
        if not os.path.isdir(exp_path): self.safe_log("Ошибка: Указанный путь не является папкой."); return
        self.all_nd2_files_in_folder = pu.get_nd2_files_sorted(exp_path)
        self.available_files_listbox.delete(0, tk.END)
        for f_name in self.all_nd2_files_in_folder: self.available_files_listbox.insert(tk.END, f_name)
        for ik_data in self.interval_data.values():
            ik_data["files"].clear()
            if ik_data["listbox"]: ik_data["listbox"].delete(0, tk.END)
        self.safe_log(f"Обновлен список .nd2 файлов из: {exp_path}")

    def add_selected_to_interval(self, interval_key):
        selected_indices = self.available_files_listbox.curselection()
        if not selected_indices: self.safe_log("Не выбраны файлы для добавления."); return
        target_lb = self.interval_data[interval_key]["listbox"]
        target_fl = self.interval_data[interval_key]["files"]
        for idx in selected_indices:
            fname = self.available_files_listbox.get(idx)
            if fname not in target_fl: target_lb.insert(tk.END, fname); target_fl.append(fname)
        self.safe_log(f"Добавлены файлы в '{self.interval_data[interval_key]['label']}'.")

    def remove_selected_from_interval(self, interval_key):
        target_lb = self.interval_data[interval_key]["listbox"]
        sel_idxs_interval = target_lb.curselection()
        if not sel_idxs_interval: self.safe_log(f"Не выбраны файлы для удаления из '{self.interval_data[interval_key]['label']}'."); return
        for idx in reversed(sel_idxs_interval):
            fname = target_lb.get(idx); target_lb.delete(idx)
            if fname in self.interval_data[interval_key]["files"]: self.interval_data[interval_key]["files"].remove(fname)
        self.safe_log(f"Удалены файлы из '{self.interval_data[interval_key]['label']}'.")

    def _validate_inputs(self):
        if not os.path.isdir(self.experiment_path.get()):
            messagebox.showerror("Ошибка", "Не выбрана корректная папка эксперимента.", parent=self.root); return False
        if not any(data["files"] for data in self.interval_data.values()):
             messagebox.showerror("Ошибка", "Ни один интервал не содержит видеофайлов.", parent=self.root); return False
        return True

    def _ask_integer_main_thread(self, title, prompt, initialvalue, minvalue):
        result = simpledialog.askinteger(title, prompt, initialvalue=initialvalue, minvalue=minvalue, parent=self.root)
        self.input_queue.put(result)

    def get_num_chunks_from_user(self, filename_short):
        while not self.input_queue.empty():
            try: self.input_queue.get_nowait()
            except queue.Empty: break
        
        if self.root and self.root.winfo_exists():
            self.root.after(0, lambda: self._ask_integer_main_thread(
                "Разделение на части",
                f"Файл {filename_short} без z-полей.\nНа сколько частей разделить? (1=не делить)",
                initialvalue=1, minvalue=1
            ))
        else: 
            self.safe_log("Главное окно закрыто, не могу запросить число чанков.")
            return 1 

        try:
            num_chunks = self.input_queue.get(timeout=60) 
            return num_chunks
        except queue.Empty:
            self.safe_log(f"Таймаут ожидания ввода для {filename_short}. Используется 1 чанк.")
            return 1 

    def run_thread_with_lock(self, target_function_name_str):
        if not self._validate_inputs(): return
        if self.processing_thread_active:
            messagebox.showwarning("Занято", "Другой процесс обработки уже запущен.", parent=self.root); return
        
        self.processing_thread_active = True
        target_method = getattr(self, target_function_name_str)
        self.active_thread = threading.Thread(target=target_method, daemon=True, name=target_function_name_str)
        self.active_thread.start()
        self.root.after(100, self.check_thread_completion)

    def check_thread_completion(self):
        if self.active_thread and self.active_thread.is_alive():
            self.root.after(100, self.check_thread_completion)
        else:
            if self.processing_thread_active: 
                 self.safe_log(f"Поток {self.active_thread.name if self.active_thread else 'обработки'} завершил работу.")
            self.processing_thread_active = False
            self.active_thread = None

    def run_interval_processing_thread(self): self.run_thread_with_lock("_execute_interval_processing")
    def run_finals_assembly_thread(self): self.run_thread_with_lock("_execute_finals_assembly")
    def run_h5_slicing_and_cellpose_thread(self): self.run_thread_with_lock("_execute_h5_slicing_and_cellpose")

    def _handle_exception_in_thread(self, func_name, e):
        print(f"ПЕРВОНАЧАЛЬНАЯ КРИТИЧЕСКАЯ ОШИБКА в {func_name}: {e}")
        traceback_str = traceback.format_exc()
        print(traceback_str)
        self.safe_log(f"КРИТИЧЕСКАЯ ОШИБКА в {func_name}: {e}")
        self.safe_log(traceback_str)
        if self.root and self.root.winfo_exists():
            self.root.after(0, lambda e_msg=str(e), fn=func_name: messagebox.showerror(
                "Критическая ошибка",
                f"Произошла ошибка в '{fn}': {e_msg}\nСм. консоль/лог для деталей.",
                parent=self.root
            ))

    def save_all_selected_cells_from_gui_data(self, all_processed_fields_data):
        self.safe_log("Сохранение результатов выбора клеток из Global GUI...")
        time_step_val = self.time_step_var.get() # Получаем актуальное значение шага времени

        for field_data in all_processed_fields_data:
            if field_data.get("selected_cells_for_this_field"): 
                pu.save_field_data_to_excel(
                    trackpy_trajectories_dict=field_data["tp_dict"],
                    original_filename=field_data["original_filename"],
                    field_or_chunk_id=field_data["field_or_chunk_id"],
                    interval_id=field_data["interval_id"],
                    current_file_global_start_time_sec=field_data["current_file_global_start_time_sec"],
                    output_path_base=field_data["output_path_base"],
                    selected_cell_info_list_with_gui_id=field_data["selected_cells_for_this_field"],
                    time_step_seconds=time_step_val
                )
            else:
                self.safe_log(f"Нет выбранных клеток для {field_data['original_filename']} - {field_data['field_or_chunk_id']}, Excel не создан.")
        self.safe_log("Сохранение результатов выбора клеток завершено.")

    def _execute_interval_processing(self):
        func_name = "_execute_interval_processing"
        try:
            self.safe_log("--- Начало обработки интервалов (Трекинг и GUI выбора клеток) ---")
            exp_path = self.experiment_path.get()
            channel = self.channel_var.get()
            time_step_original = self.time_step_var.get()
            pu.DEFAULT_TIME_STEP_SECONDS = time_step_original # Устанавливаем глобально для utils
            tp_params = {"diameter": self.tp_diameter_var.get(), "link_range": self.tp_link_range_var.get(), 
                         "memory": self.tp_memory_var.get(), "min_len": self.tp_min_len_var.get()}
            base_output_folder_for_excel_fields = os.path.join(exp_path, f"{self.patient_name_var.get()}_processing_results")
            os.makedirs(base_output_folder_for_excel_fields, exist_ok=True)

            all_processed_fields_data = [] # Для сбора данных для глобального GUI
            cumulative_time_offset_for_experiment = 0.0
            sorted_interval_keys = sorted(self.interval_data.keys(), key=lambda k: self.interval_data[k]["offset_sec"])

            for interval_key in sorted_interval_keys:
                data = self.interval_data[interval_key]
                interval_folder_id = data["id_for_folder"]
                if not data["files"]: 
                    self.safe_log(f"Пропуск интервала '{data['label']}' - нет файлов.")
                    continue
                self.safe_log(f"Обработка интервала: {data['label']}")

                for file_idx, filename_short in enumerate(data["files"]):
                    if not self.processing_thread_active: 
                        self.safe_log(f"{func_name} прерван.")
                        return
                    filepath_full = os.path.join(exp_path, filename_short)
                    self.safe_log(f"  Файл: {filename_short}")
                    current_file_global_start_time = cumulative_time_offset_for_experiment
                    self.safe_log(f"    Глобальное время начала файла: {current_file_global_start_time:.2f} сек")

                    z_stacks_count, _ = pu.get_z_size_nd2(filepath_full)
                    tasks_for_processing = []
                    if z_stacks_count is not None and z_stacks_count > 0:
                        self.safe_log(f"    Найдено {z_stacks_count} полей (z-stacks).")
                        for z_idx in range(z_stacks_count): 
                            tasks_for_processing.append({"z_idx": z_idx, "chunk_idx": None, "max_chunks": None, "field_id": f"z{z_idx}"})
                    else:
                        num_chunks = self.get_num_chunks_from_user(filename_short)
                        if num_chunks is None: 
                            self.safe_log(f"    Обработка {filename_short} пропущена (нет ввода).")
                            continue
                        if num_chunks > 1:
                            self.safe_log(f"    Файл будет разделен на {num_chunks} частей.")
                            for chunk_i in range(num_chunks): 
                                tasks_for_processing.append({"z_idx": None, "chunk_idx": chunk_i, "max_chunks": num_chunks, "field_id": f"chunk{chunk_i}"})
                        else:
                            self.safe_log(f"    Файл обрабатывается целиком.")
                            tasks_for_processing.append({"z_idx": None, "chunk_idx": 0, "max_chunks": 1, "field_id": "full"})
                    
                    file_duration_this_file_for_offset_calc = 0
                    for task_def in tasks_for_processing:
                        if not self.processing_thread_active: 
                            self.safe_log(f"{func_name} прерван.")
                            return
                        self.safe_log(f"      Обработка: {filename_short} - {task_def['field_id']}")
                        
                        img_seq, times_local = pu.load_nd2_data(filepath_full, task_def["z_idx"], channel, 
                                                                task_def["max_chunks"], task_def["chunk_idx"])
                        
                        if img_seq is None: 
                            self.safe_log(f"        Не удалось загрузить данные для {filename_short} - {task_def['field_id']}. Пропуск.")
                            if file_duration_this_file_for_offset_calc == 0 and times_local is not None and len(times_local) > 0:
                                file_duration_this_file_for_offset_calc = times_local[-1] + time_step_original
                            continue
                        
                        current_num_frames = len(img_seq)

                        if file_duration_this_file_for_offset_calc == 0 and times_local is not None and len(times_local) > 0:
                            file_duration_this_file_for_offset_calc = times_local[-1] + time_step_original

                        tp_df, tp_dict = pu.process_frames_trackpy(img_seq, **tp_params)
                        
                        del img_seq # Освобождаем память от сырых изображений

                        if tp_df.empty: 
                            self.safe_log(f"        Трекинг не дал результатов для {filename_short} - {task_def['field_id']}. Пропуск.")
                            continue 
                        
                        field_data_for_gui = {
                            "load_params": {
                                "filepath_full": filepath_full,
                                "z_idx": task_def["z_idx"],
                                "channel": channel,
                                "max_chunks": task_def["max_chunks"],
                                "chunk_idx": task_def["chunk_idx"]
                            },
                            "num_frames": current_num_frames,
                            "tp_df": tp_df,
                            "tp_dict": tp_dict,
                            "original_filename": filename_short,
                            "field_or_chunk_id": task_def['field_id'],
                            "interval_id": interval_folder_id,
                            "interval_name": data['label'],
                            "current_file_global_start_time_sec": current_file_global_start_time,
                            "output_path_base": base_output_folder_for_excel_fields,
                            "selected_cells_for_this_field": [],
                        }
                        all_processed_fields_data.append(field_data_for_gui)
                        self.safe_log(f"        Данные для {filename_short} - {task_def['field_id']} подготовлены для GUI.")

                    cumulative_time_offset_for_experiment += file_duration_this_file_for_offset_calc
                    self.safe_log(f"    Завершен файл {filename_short}. Следующий файл начнется с ~{cumulative_time_offset_for_experiment:.2f} сек.")
            
            # После всех циклов трекинга:
            if not all_processed_fields_data:
                self.safe_log("Обработка завершена, но нет данных для отображения в GUI выбора клеток.")
                if self.root and self.root.winfo_exists():
                    self.root.after(0, lambda: messagebox.showinfo("Завершено", "Трекинг завершен, но не найдено траекторий для выбора клеток.", parent=self.root))
                return

            self.safe_log("Запуск глобального GUI выбора клеток...")
            if self.root and self.root.winfo_exists():
                # Передаем метод safe_log из PipelineApp в GUI для логирования
                global_gui = pu.GlobalCellSelectorGUI(all_processed_fields_data, parent_app_logger=self.safe_log)

                if global_gui.fig is None: # Если GUI не смог инициализироваться (например, нет данных)
                    self.safe_log("GlobalCellSelectorGUI не был инициализирован (возможно, нет данных).")
                else:
                    while plt.fignum_exists(global_gui.fig.number):
                        if not (self.root and self.root.winfo_exists()):
                            self.safe_log("Главное окно Tkinter закрыто во время работы GlobalCellSelectorGUI. Прерывание.")
                            plt.close(global_gui.fig)
                            return 
                        self.root.update_idletasks()
                        import time; time.sleep(0.1) # Даем Tkinter и Matplotlib время на обработку событий
                
                # После закрытия GUI, вызываем сохранение данных, которые GUI накопил
                self.save_all_selected_cells_from_gui_data(all_processed_fields_data)
                
                self.safe_log(f"--- {func_name} (включая выбор клеток) завершен успешно ---")
                if self.root and self.root.winfo_exists():
                     self.root.after(0, lambda: messagebox.showinfo("Завершено", f"{func_name} и выбор клеток завершены успешно.", parent=self.root))
            else:
                self.safe_log("Главное окно было закрыто перед запуском GlobalCellSelectorGUI.")
                return

        except Exception as e:
            self._handle_exception_in_thread(func_name, e)
        finally:
             if threading.current_thread() is self.active_thread:
                self.processing_thread_active = False


    def _execute_finals_assembly(self):
        func_name = "_execute_finals_assembly"
        try:
            self.safe_log("--- Начало сборки итоговых 'finals' Excel файлов ---")
            exp_path = self.experiment_path.get(); patient_name_var_val = self.patient_name_var.get()
            base_results_folder = os.path.join(exp_path, f"{patient_name_var_val}_processing_results")
            if not os.path.isdir(base_results_folder):
                self.safe_log(f"Ошибка: Папка промежуточных результатов ({base_results_folder}) не найдена.")
                if self.root and self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Папка {base_results_folder} не найдена.", parent=self.root))
                return

            folders_with_excel_field_files = []
            for data_item in self.interval_data.values():
                interval_res_folder = os.path.join(base_results_folder, f"interval_{data_item['id_for_folder']}_results")
                if os.path.isdir(interval_res_folder): folders_with_excel_field_files.append(interval_res_folder)
            
            if not folders_with_excel_field_files:
                self.safe_log(f"Ошибка: Не найдено подпапок 'interval_X_results' в {base_results_folder}.")
                if self.root and self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showerror("Ошибка", "Не найдено папок с результатами полей.", parent=self.root))
                return

            source_file_prefixes = set()
            for interval_folder in folders_with_excel_field_files:
                for fname in os.listdir(interval_folder):
                    if fname.lower().endswith(".xlsx") and "_field" in fname: source_file_prefixes.add(fname.split("_field")[0])
            
            if not source_file_prefixes:
                self.safe_log("Не найдено Excel файлов полей (_field*.xlsx) для агрегации.")
                if self.root and self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showerror("Ошибка", "Не найдено Excel файлов полей.", parent=self.root))
                return
            self.safe_log(f"Обнаружены префиксы исходных файлов для агрегации: {source_file_prefixes}")

            output_folder_for_final_excel = os.path.join(exp_path, f"{patient_name_var_val}_final_aggregated_data")
            
            for src_prefix in source_file_prefixes:
                if not self.processing_thread_active: self.safe_log(f"{func_name} прерван."); return
                self.safe_log(f"  Агрегация для префикса: {src_prefix}")
                current_final_basename_for_type = f"{patient_name_var_val}_{src_prefix}"
                pu.aggregate_and_save_final_data_v2(folders_with_excel_field_files, src_prefix, current_final_basename_for_type, output_folder_for_final_excel)
            
            self.safe_log(f"--- {func_name} завершен успешно ---")
            if self.root and self.root.winfo_exists():
                self.root.after(0, lambda: messagebox.showinfo("Завершено", f"{func_name} завершен успешно.", parent=self.root))
        except Exception as e:
            self._handle_exception_in_thread(func_name, e)
        finally:
            if threading.current_thread() is self.active_thread: self.processing_thread_active = False


    def _execute_h5_slicing_and_cellpose(self):
        func_name = "_execute_h5_slicing_and_cellpose"
        try:
            self.safe_log("--- Начало нарезки H5 и сегментации Cellpose ---")
            exp_path = self.experiment_path.get(); patient_name = self.patient_name_var.get()
            channel = self.channel_var.get(); time_step = self.time_step_var.get()
            try:
                target_times_sec_list = [float(t.strip()) for t in self.h5_times_str_var.get().split(',') if t.strip()]
                if not target_times_sec_list: 
                    self.safe_log("Ошибка: не указаны времена для H5.")
                    if self.root and self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showerror("Ошибка", "Укажите времена для H5.", parent=self.root))
                    return
            except ValueError: 
                self.safe_log("Ошибка: неверный формат времен для H5.")
                if self.root and self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showerror("Ошибка", "Неверный формат времен для H5.", parent=self.root))
                return

            all_experiment_video_files_ordered_shortnames = []
            sorted_interval_keys = sorted(self.interval_data.keys(), key=lambda k: self.interval_data[k]["offset_sec"])
            for interval_key in sorted_interval_keys: all_experiment_video_files_ordered_shortnames.extend(self.interval_data[interval_key]["files"])
            
            if not all_experiment_video_files_ordered_shortnames:
                self.safe_log("Нет видеофайлов для нарезки H5.")
                if self.root and self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showerror("Ошибка", "Нет видеофайлов в интервалах для H5.", parent=self.root))
                return
            self.safe_log(f"Порядок видеофайлов для H5: {all_experiment_video_files_ordered_shortnames}")

            output_h5_folder = os.path.join(exp_path, f"{patient_name}_h5_and_cellpose_results")
            os.makedirs(output_h5_folder, exist_ok=True)
            cp_model = self.cp_model_type_var.get(); cp_diam_val = self.cp_diameter_var.get()
            cp_diam = cp_diam_val if cp_diam_val > 0 else None

            for t_sec in target_times_sec_list:
                if not self.processing_thread_active: self.safe_log(f"{func_name} прерван."); return
                self.safe_log(f"  Нарезка кадра для времени: {t_sec} сек")
                h5_file_path = pu.save_frame_as_h5_tiff(t_sec, exp_path, all_experiment_video_files_ordered_shortnames,
                                                     time_step, channel, patient_name, output_folder_override=output_h5_folder)
                if h5_file_path:
                    self.safe_log(f"    H5 файл: {h5_file_path}\n    Запуск Cellpose...")
                    seg_res = pu.run_cellpose_segmentation(h5_file_path, cp_model, cp_diam, output_h5_folder)
                    if seg_res: self.safe_log(f"    Сегментация завершена. Маска: {os.path.basename(seg_res[0])}")
                    else: self.safe_log(f"    Ошибка сегментации Cellpose.")
                else: self.safe_log(f"    Не удалось сохранить H5 для {t_sec} сек.")
            
            self.safe_log(f"--- {func_name} завершен успешно ---")
            if self.root and self.root.winfo_exists():
                self.root.after(0, lambda: messagebox.showinfo("Завершено", f"{func_name} завершен успешно.", parent=self.root))
        except Exception as e:
            self._handle_exception_in_thread(func_name, e)
        finally:
            if threading.current_thread() is self.active_thread: self.processing_thread_active = False


if __name__ == "__main__":
    root_tk = tk.Tk()
    app = PipelineApp(root_tk)
    root_tk.mainloop()