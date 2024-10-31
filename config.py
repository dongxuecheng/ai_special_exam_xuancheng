import numpy as np

SERVER_IP='192.168.10.109'
SERVER_PORT_WELDING_K2 = '5002'  # 焊接k2服务器端口

SAVE_IMG_PATH_WELDING_K2 = 'images/welding/k2'  # 焊接图片保存在服务器的实际位置




URL_IMG_PATH_WELDING_K2 = f'http://{SERVER_IP}:{SERVER_PORT_WELDING_K2}/{SAVE_IMG_PATH_WELDING_K2}'  # 通过端口映射能够访问的位置 焊接考核科目2



RTSP_WELDING_WEARING='rtsp://admin:yaoan1234@192.168.10.221/cam/realmonitor?channel=1&subtype=0'#焊接的穿戴
RTSP_WELDING_DESK='rtsp://admin:yaoan1234@192.168.10.222/cam/realmonitor?channel=1&subtype=0'#焊接的焊台
RTSP_WELDING_MACHINE_SWITCH='rtsp://admin:yaoan1234@192.168.10.223/cam/realmonitor?channel=1&subtype=0'#焊接的焊机开关
RTSP_WELDING_MAIN_SWITCH_AND_OIL_TANK='rtsp://admin:yaoan1234@192.168.10.224/cam/realmonitor?channel=1&subtype=0'#焊接的总开关和油桶

WEIGHTS_WELDING_DESK_CLS='weights/welding_desk_cls_1030.pt'
WEIGHTS_WELDING_DESK_DETECT='weights/welding_desk_detect_1030.pt'
WEIGHTS_WELDING_MACHINE_SWITCH_DETECT='weights/welding_machine_switch_detect_1030.pt'
WELDING_MAIN_SWITCH_GLOVES_OIL_TANK_DETECT='weights/welding_main_switch_gloves_oil_tank_detect_1030.pt'

VIDEOS_WELDING=[
    RTSP_WELDING_MACHINE_SWITCH,
    RTSP_WELDING_MAIN_SWITCH_AND_OIL_TANK,
    RTSP_WELDING_DESK
]

WEIGHTS_WELDING_RESET=[
    WEIGHTS_WELDING_MACHINE_SWITCH_DETECT,
    WELDING_MAIN_SWITCH_GLOVES_OIL_TANK_DETECT,
    WEIGHTS_WELDING_DESK_DETECT,
]

WEIGHTS_WELDING_EXAM=[
    WEIGHTS_WELDING_MACHINE_SWITCH_DETECT,
    WELDING_MAIN_SWITCH_GLOVES_OIL_TANK_DETECT,
    WEIGHTS_WELDING_DESK_DETECT,
    WEIGHTS_WELDING_DESK_CLS
]

WELDING_REGION1 = np.array([[462, 279], [566, 914], [763, 1008], [639, 213]], np.int32)#搭铁线的识别区域
WELDING_REGION2 = np.array([[650, 220], [770, 1005], [1462, 720], [1420, 181]], np.int32)#放置焊件的识别区域