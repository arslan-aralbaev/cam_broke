import sys
import cv2
import numpy as np
import pyvirtualcam
import datetime
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, \
    QLabel, QTabWidget, QSlider, QCheckBox, QFileDialog,\
    QLineEdit, QComboBox, QMessageBox
import sqlite3


capture = cv2.VideoCapture(0)  # подключение камеры
capture.set(cv2.CAP_PROP_FPS, 60)  # Чистота кадров

# каскады
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# определение точного времени
now = datetime.datetime.now()
geek_now = now.strftime("%d-%m-%Y %H:%M:%S")

# обЪявление бд
base = sqlite3.connect('unknown_data.db')
cursor = base.cursor()
cursor.execute(
    'CREATE TABLE IF NOT EXISTS data(date TEXT, data TEXT);')
base.commit()


def data_base(data_nw):
    # сохранение логов
    sf_time = now.strftime("%d-%m-%Y %H:%M:%S")
    # print(sf_time, data_nw) дле дебага
    cursor.execute(f'''insert into data(date, data) values(?, ?);''', (sf_time, str(data_nw)))
    base.commit()


def blur_ef(img, blurs):
    #  содание блюр эффекта
    (h, w) = img.shape[:2]
    dw = int(w / 3.0)
    dh = int(h / 3.0)
    if dw % 2 == 0:
        dw -= 1
    if dh % 2 == 0:
        dh -= 1
    return cv2.GaussianBlur(img, (dh, dw), blurs)


def shift_img(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy > 0:
        img[:dy, :] = 0
    elif dy < 0:
        img[dy:, :] = 0
    if dx > 0:
        img[:, :dx] = 0
    elif dx < 0:
        img[:, dx:] = 0
    return img


def glif(img, fl):
    #  наложение различных эффектов
    color = cv2.COLORMAP_SUMMER
    if fl == 'summer':
        color = cv2.COLORMAP_SUMMER
    elif fl == 'spring':
        color = cv2.COLORMAP_SPRING
    elif fl == 'winter':
        color = cv2.COLORMAP_WINTER
    elif fl == 'autumn':
        color = cv2.COLORMAP_AUTUMN
    elif fl == 'jet':
        color = cv2.COLORMAP_JET
    elif fl == 'ocean':
        color = cv2.COLORMAP_OCEAN
    elif fl == 'cool':
        color = cv2.COLORMAP_COOL
    elif fl == 'hsv':
        color = cv2.COLORMAP_HSV
    elif fl == 'pink':
        color = cv2.COLORMAP_PINK
    elif fl == 'plasma':
        color = cv2.COLORMAP_PLASMA
    elif fl == 'viridis':
        color = cv2.COLORMAP_VIRIDIS
    elif fl == 'turbo':
        color = cv2.COLORMAP_TURBO
    elif fl == 'cividis':
        color = cv2.COLORMAP_CIVIDIS
    elif fl == 'deep green':
        color = cv2.COLORMAP_DEEPGREEN
    imgw = cv2.applyColorMap(img, color)
    # imgw = img
    bl, bg = 2, 3
    for y in range(imgw.shape[0]):
        if y % (bl + bg) < bl:
            imgw[y, :, :] = imgw[y, :, :] * np.random.uniform(0.1, 0.3)
    # создание глиф эффекта
    imgw = cv2.addWeighted(imgw, 0.2, shift_img(imgw.copy(), 5, 5), 0.8, 0)
    imgw = cv2.addWeighted(imgw, 0.4, shift_img(imgw.copy(), -5, -5), 0.6, 0)
    imgw = cv2.addWeighted(img, 0.5, imgw, 0.6, 0)
    return imgw


# класс работы с cv2
class Maisyc:
    def __init__(self, path, blur, conturs, blur_sost, face_bool, eye_bool, mouth_bool, w, h, fps, filters):
        self.path = path
        self.blur = blur
        self.conturs = conturs
        self.blur_sost = blur_sost  # face, eye, mouth
        self.face_bool = face_bool
        self.eye_bool = eye_bool
        self.mouth_bool = mouth_bool
        self.w, self.h = w, h
        self.fps = fps
        self.filters = filters

    def image_cin(self):
        fmt = pyvirtualcam.PixelFormat.BGR
        with pyvirtualcam.Camera(width=self.w, height=self.h, fps=self.fps, fmt=fmt) as cam:
            print(f'Using virtual camera: {cam.device}')
            data_base(f'Using virtual camera: {cam.device}')
            while True:
                ret, img = capture.read()  # чтение иизображения
                # поиск по каскадам
                faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(28, 20))
                eyes = eyes_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10))
                mouth = mouth_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=25, minSize=(28, 20))
                # работа с каскадом и изображением лица
                if self.face_bool:
                    for (x, y, w, h) in faces:
                        if self.conturs[0]:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if self.blur_sost[0]:
                            data_base('open: face cascade')
                            if self.blur_sost[3]:
                                img[y:y+h, x:x+w] = glif(blur_ef(img[y:y+h, x:x+w], self.blur[0]), self.filters)
                            else:
                                img[y:y + h, x:x + w] = blur_ef(img[y:y + h, x:x + w], self.blur[0])

                if self.eye_bool:
                    for (x, y, w, h) in eyes:
                        if self.conturs[1]:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        if self.blur_sost[1]:
                            data_base('open: eye cascade')
                            if self.blur_sost[3]:
                                img[y:y+h, x:x+w] = glif(blur_ef(img[y:y+h, x:x+w], self.blur[1]), self.filters)
                            else:
                                img[y:y + h, x:x + w] = blur_ef(img[y:y + h, x:x + w], self.blur[1])

                if self.mouth_bool:
                    for (x, y, w, h) in mouth:
                        if self.conturs[2]:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        if self.blur_sost[2]:
                            data_base('open: mouth cascade')
                            if self.blur_sost[3]:
                                img[y:y + h, x:x + w] = glif(blur_ef(img[y:y + h, x:x + w], self.blur[2]), self.filters)
                            else:
                                img[y:y + h, x:x + w] = blur_ef(img[y:y + h, x:x + w], self.blur[2])
                image_sost = cv2.resize(img, (self.w, self.h))
                cv2.imshow('camera_0', img[:, ::-1])
                cam.send(image_sost)
                cam.sleep_until_next_frame()
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    print('close')
                    data_base('close: all cascades')
                    break
                elif key == 46:
                    cv2.imwrite(self.path, image_sost)
                    print(f'saved as {self.path}')
                    data_base(f'saved as {self.path}')


class Focus(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(300, 300, 535, 375)
        self.setWindowTitle('Cam crash')
        self.setStyleSheet("background-color: rgb(150, 150, 150);")

        self.tabWidget = QTabWidget(self)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 521, 361))
        self.tabWidget.setMovable(True)
        self.tabWidget.setObjectName("tabWidget")

        self.main_tab = QtWidgets.QWidget()  # Main or Menu
        self.main_tab.setObjectName("tab")
        self.main_tab.setStyleSheet('''
                                    background-color: rgb(210, 210, 210);
                                    font: 75 "Terminal";
                                    ''')
        self.btn = QPushButton(self.main_tab)
        self.btn.setText('▶')
        self.btn.move(150, 150)
        self.btn.setToolTip("Запуск инструмента")
        self.exo = QPushButton(self.main_tab)
        self.exo.setText('exit')
        self.exo.move(150, 175)
        self.exo.setToolTip("Выход из программы")
        self.user_settings = QPushButton(self.main_tab)
        self.user_settings.setText('}{')
        self.user_settings.move(150, 225)
        self.user_settings.setToolTip("Запуск инструмента редактирования")

        # self.user_settings.clicked.connect()
        self.exo.clicked.connect(self.nevn)
        self.btn.clicked.connect(self.image_vin)

        self.ik = QPushButton(self.main_tab)
        self.ik.setText('About')
        self.exo.setToolTip("Нажми :)")
        self.ik.move(150, 200)
        self.ik.clicked.connect(self.info)

        self.main_label = QLabel(self.main_tab)
        self.main_label.setText('Center')
        self.main_label.move(360, 15)
        self.main_label.setStyleSheet('''
                    font: 75 12pt "Terminal";
                    text - decoration: underline;
                    color: rgb(255, 127, 70);''')
        self.tabWidget.addTab(self.main_tab, "Main")

        self.face_settings = QtWidgets.QWidget()
        self.face_settings.setObjectName("face settings")
        self.face_settings.setStyleSheet('''
                                        background-color: rgb(255, 255, 255);
                                        font: 75 "Terminal";
                                        ''')
        self.tabWidget.addTab(self.face_settings, "Face settings")
        self.face_check = QCheckBox(self.face_settings)
        self.face_check.setText('Наличие маски на всё лицо:')
        self.face_check.move(15, 15)
        self.face_check.setCheckState(2)
        self.face_check.setToolTip("Включение функции отслеживания лица")

        self.contur_check_box_face = QCheckBox(self.face_settings)
        self.contur_check_box_face.setText('Contur effect')
        self.contur_check_box_face.setToolTip("Включение контуров лица")
        self.contur_check_box_face.move(15, 35)
        self.blured_check_box_face = QCheckBox(self.face_settings)
        self.blured_check_box_face.setText('Blur effect')
        self.blured_check_box_face.setToolTip("Включение блюр эффекта лица")
        self.blured_check_box_face.move(15, 55)

        self.blured_label = QLabel(self.face_settings)
        self.blured_label.setText('Blure')
        self.blured_label.setToolTip("Параметры блюр еффекта")
        self.blured_label.move(15, 75)

        self.blured_face_slider = QSlider(self.face_settings)
        self.blured_face_slider.setOrientation(1)
        self.blured_face_slider.move(50, 75)

        self.main_face_label = QLabel(self.face_settings)
        self.main_face_label.setText('Face settings')
        self.main_face_label.move(360, 15)
        self.main_face_label.setStyleSheet('''
            font: 75 12pt "Terminal";
            text - decoration: underline;
            color: rgb(0, 255, 127);''')
        self.tabWidget.addTab(self.face_settings, "Face settings")

        # Eye Tab
        self.eye_settings = QtWidgets.QWidget()
        self.eye_settings.setObjectName("eye settings")
        self.eye_settings.setStyleSheet('''
                                        background-color: rgb(255, 255, 255);
                                        font: 75 "Terminal";
                                        
                                        ''')
        self.tabWidget.addTab(self.eye_settings, "Eye settings")
        self.eye_check = QCheckBox(self.eye_settings)
        self.eye_check.setText('Наличие маски на глаза:')
        self.eye_check.move(15, 15)
        self.eye_check.setCheckState(2)
        self.eye_check.setToolTip("Включение функции отслеживания глаз")

        self.contur_check_box_eye = QCheckBox(self.eye_settings)
        self.contur_check_box_eye.setText('Contur effect')
        self.contur_check_box_eye.move(15, 35)
        self.contur_check_box_eye.setToolTip("Включение контуров глаз")
        self.blured_check_box_eye = QCheckBox(self.eye_settings)
        self.blured_check_box_eye.setText('Blur effect')
        self.blured_check_box_eye.setToolTip("Включение блюр эффекта глаз")
        self.blured_check_box_eye.move(15, 55)

        self.blured_label = QLabel(self.eye_settings)
        self.blured_label.setText('Blure')
        self.blured_label.setToolTip("Параметры блюр еффекта")
        self.blured_label.move(15, 75)

        self.blured_eye_slider = QSlider(self.eye_settings)
        self.blured_eye_slider.setOrientation(1)
        self.blured_eye_slider.move(50, 75)

        self.main_face_label = QLabel(self.eye_settings)
        self.main_face_label.setText('Eyes settings')
        self.main_face_label.move(360, 15)
        self.main_face_label.setStyleSheet('''
                    font: 75 12pt "Terminal";
                    text - decoration: underline;
                    color: rgb(255, 85, 127);''')

        # Mouth Tab
        self.mouth_settings = QtWidgets.QWidget()
        self.mouth_settings.setObjectName("mouth settings")
        self.mouth_settings.setStyleSheet('''
                                            background-color: rgb(255, 255, 255);
                                            font: 75 "Terminal";
                                            ''')
        self.tabWidget.addTab(self.mouth_settings, "Mouth settings")
        self.mouth_check = QCheckBox(self.mouth_settings)
        self.mouth_check.setText('Наличие маски на рот:')
        self.mouth_check.move(15, 15)
        self.mouth_check.setCheckState(0)
        self.mouth_check.setToolTip("Включение функции отслеживания рта")

        self.contur_check_box_mouth = QCheckBox(self.mouth_settings)
        self.contur_check_box_mouth.setText('Contur effect')
        self.contur_check_box_mouth.move(15, 35)
        self.contur_check_box_mouth.setToolTip("Включение контуров рта")
        self.blured_check_box_mouth = QCheckBox(self.mouth_settings)
        self.blured_check_box_mouth.setText('Blur effect')
        self.blured_check_box_mouth.move(15, 55)
        self.blured_check_box_mouth.setToolTip("Включение блюр эффекта рта")

        self.blured_label = QLabel(self.mouth_settings)
        self.blured_label.setText('Blure')
        self.blured_label.move(15, 75)
        self.blured_label.setToolTip("Параметры блюр еффекта")

        self.blured_mouth_slider = QSlider(self.mouth_settings)
        self.blured_mouth_slider.setOrientation(True)
        self.blured_mouth_slider.move(50, 75)

        self.main_face_label = QLabel(self.mouth_settings)
        self.main_face_label.setText('Mouths setting')
        self.main_face_label.move(360, 15)
        self.main_face_label.setStyleSheet('''
                            font: 75 12pt "Terminal";
                            text - decoration: underline;
                            color: rgb(85, 170, 255);''')
        self.main_face_label = QLabel(self.mouth_settings)
        self.main_face_label.setText('''⚠⚠⚠Warning! 
It is not recommended
to use these settings.
The version of these settings is unstable and
may cause critical errors.⚠⚠⚠''')
        self.main_face_label.move(15, 220)
        self.main_face_label.setStyleSheet('''
                                            font: 75 "Terminal";
                                            text - decoration: underline;
                                            color: rgb(255, 0, 0);''')
        # Another Settings Tab
        self.setting_tab = QtWidgets.QWidget()
        self.setting_tab.setObjectName("tab_2")
        self.setting_tab.setStyleSheet('''
                                    background-color: rgb(255, 255, 255);
                                    font: 75 "Terminal";
                                    ''')
        self.glif = QCheckBox(self.setting_tab)
        self.glif.setText('Glif effect:')
        self.glif.move(15, 15)
        self.glif.setCheckState(False)
        self.glif.setToolTip("Включение глиф эффекта")

        self.weight_inp_label = QLabel(self.setting_tab)
        self.weight_inp_label.setText('Screen Size w: ')
        self.weight_inp_label.move(15, 45)
        self.weight_inp_label.setToolTip("Параметры разрешения изображения")
        self.weight_inp = QLineEdit(self.setting_tab)
        self.weight_inp.resize(40, 20)
        self.weight_inp.move(110, 45)
        self.weight_inp.setText('1280')

        self.hight_inp_label = QLabel(self.setting_tab)
        self.hight_inp_label.setText('h: ')
        self.hight_inp_label.move(160, 45)
        self.hight_inp = QLineEdit(self.setting_tab)
        self.hight_inp.resize(40, 20)
        self.hight_inp.move(180, 45)
        self.hight_inp.setText('720')

        self.FPS_label = QLabel(self.setting_tab)
        self.FPS_label.setText('FPS: ')
        self.FPS_label.move(15, 70)
        self.FPS_label.setToolTip("Настройка частоты кадров")
        self.FPS_inp = QLineEdit(self.setting_tab)
        self.FPS_inp.resize(40, 20)
        self.FPS_inp.move(50, 70)
        self.FPS_inp.setText('20')

        self.filters_label = QLabel(self.setting_tab)
        self.filters_label.setText('Filters: ')
        self.filters_label.move(15, 95)
        self.filters_label.setToolTip("Список дополнительных еффектов для глифа")
        self.filters = QComboBox(self.setting_tab)
        self.filters.move(60, 95)
        self.filters.addItem('summer')
        self.filters.addItem('winter')
        self.filters.addItem('spring')
        self.filters.addItem('autumn')
        self.filters.addItem('jet')
        self.filters.addItem('ocean')
        self.filters.addItem('cool')
        self.filters.addItem('hsv')
        self.filters.addItem('pink')
        self.filters.addItem('plasma')
        self.filters.addItem('viridis')
        self.filters.addItem('turbo')
        self.filters.addItem('cividis')
        self.filters.addItem('deep green')

        self.main_face_label = QLabel(self.setting_tab)
        self.main_face_label.setText('Another setting')
        self.main_face_label.move(360, 15)
        self.main_face_label.setStyleSheet('''
                                    font: 75 12pt "Terminal";
                                    text - decoration: underline;
                                    color: rgb(170, 85, 255);''')
        self.paths = QPushButton(self.setting_tab)
        self.paths.setText('C:/')
        self.paths.move(15, 115)
        self.paths.setToolTip("Нажмите и выберите папку для сохранения фото(фотка сохраняется по кнопке 'ю')")
        self.paths.clicked.connect(self.files_save)

        self.photo_name_label = QLabel(self.setting_tab)
        self.photo_name_label.setText('File name: ')
        self.photo_name_label.move(15, 140)
        self.photo_name_label.setToolTip("Имя сохраняемого файла")
        self.photo_name = QLineEdit(self.setting_tab)
        self.photo_name.resize(40, 20)
        self.photo_name.move(110, 140)
        self.photo_name.setText('milafka')
        self.tabWidget.addTab(self.setting_tab, "Another settings")
        self.readlogs = QPushButton(self.setting_tab)
        self.readlogs.setText('read logs')
        self.readlogs.move(15, 165)
        self.readlogs.setToolTip('Что последнее было сделано')
        self.readlogs.clicked.connect(self.r_log)

    def r_log(self):
        hl = QMessageBox()
        hl.setWindowTitle('Last Log')
        log = cursor.execute("""
SELECT data FROM data 
""").fetchall()
        hl.setText(str(log[-1][0]))
        hl.setIcon(QMessageBox.Information)
        hl.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        hl.exec_()

    def info(self):
        inf = QMessageBox()
        inf.setWindowTitle('About app')
        inf.setText('''Dev by Arslan A A.
Верcия: 0.2.1
Поддержка: arslan.aralbaev@yandex.com''')
        inf.setIcon(QMessageBox.Information)
        inf.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        inf.exec_()

    def nevn(self):
        print('exit')
        exit()

    def file_load(self):
        name = QApplication.instance().sender()
        name.setObjectName(QFileDialog.getOpenFileName(self.xzc_tab, 'path', '*.py')[0])

    def files_save(self):
        name = QApplication.instance().sender()
        name.setText(QFileDialog.getExistingDirectory(self.setting_tab, 'path', '.'))

    def image_vin(self):
        try:
            self.blured_count = [self.blured_face_slider.value(),
                                 self.blured_eye_slider.value(),
                                 self.blured_mouth_slider.value()]
            self.conturs = [self.contur_check_box_face.isChecked(),
                            self.contur_check_box_eye.isChecked(),
                            self.contur_check_box_mouth.isChecked()]
            print(self.blured_count)
            data_base(self.blured_count)

            Maisyc(self.paths.text() + '/' + self.photo_name.text() + '.jpg', self.blured_count, self.conturs,
                   [self.blured_check_box_face.isChecked(),
                    self.blured_check_box_eye.isChecked(),
                    self.blured_check_box_mouth.isChecked(),
                    self.glif.isChecked()], self.face_check.isChecked(),
                   self.eye_check.isChecked(), self.mouth_check.isChecked(),
                   int(self.weight_inp.text()), int(self.hight_inp.text()),
                   int(self.FPS_inp.text()), self.filters.currentText()).image_cin()
            # capture.release()
            cv2.destroyAllWindows()

        except Exception as ex:
            error = QMessageBox()
            error.setWindowTitle('Warning')
            error.setText(f'{ex}')
            error.setIcon(QMessageBox.Warning)
            error.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            error.exec_()
            print('error', ex)


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Focus()
    ex.show()
    sys.excepthook = except_hook
    sys.exit(app.exec_())
