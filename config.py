import numpy as np

SERVER_IP='192.168.10.109'

#焊接考核
SERVER_PORT_WELDING_K2 = '5002'  # 焊接k2服务器端口

SAVE_IMG_PATH_WELDING_K2 = 'images/welding/k2'  # 焊接图片保存在服务器的实际位置

URL_IMG_PATH_WELDING_K2 = f'http://{SERVER_IP}:{SERVER_PORT_WELDING_K2}/{SAVE_IMG_PATH_WELDING_K2}'  # 通过端口映射能够访问的位置 焊接考核科目2



RTSP_WELDING_DESK='rtsp://admin:yaoan1234@192.168.10.222/cam/realmonitor?channel=1&subtype=0'#焊接的焊台
RTSP_WELDING_MACHINE_SWITCH='rtsp://admin:yaoan1234@192.168.10.223/cam/realmonitor?channel=1&subtype=0'#焊接的焊机开关
RTSP_WELDING_MAIN_SWITCH_AND_OIL_TANK='rtsp://admin:yaoan1234@192.168.10.224/cam/realmonitor?channel=1&subtype=0'#焊接的总开关和油桶
RTSP_WELDING_ROOOM_CLEANING="rtsp://admin:yaoan1234@192.168.10.225/cam/realmonitor?channel=1&subtype=0"#焊接检测扫把

WEIGHTS_WELDING_DESK_CLS='weights/welding_desk_cls_1121.pt'
WEIGHTS_WELDING_DESK_DETECT='weights/welding/1127/welding_desk_detect_1127.pt'
WEIGHTS_WELDING_MACHINE_SWITCH_DETECT='weights/welding_machine_switch_detect_1030.pt'
WEIGHTS_MAIN_SWITCH_GLOVES_OIL_TANK_DETECT='weights/welding/1127/welding_main_switch_1127.pt'
WEIGHTS_BROOM_DETECT='weights/welding/1127/welding_broom_1127.pt'

VIDEOS_WELDING=[
    RTSP_WELDING_ROOOM_CLEANING,
    RTSP_WELDING_MACHINE_SWITCH,
    RTSP_WELDING_MAIN_SWITCH_AND_OIL_TANK,
    RTSP_WELDING_DESK    
]

WEIGHTS_WELDING=[
    WEIGHTS_BROOM_DETECT,
    WEIGHTS_WELDING_MACHINE_SWITCH_DETECT,
    WEIGHTS_MAIN_SWITCH_GLOVES_OIL_TANK_DETECT,
    WEIGHTS_WELDING_DESK_DETECT,
    WEIGHTS_WELDING_DESK_CLS
    
]

WELDING_REGION1 = np.array([[738, 201], [403, 317], [466, 848], [769, 992],[854,978]], np.int32)#搭铁线的识别区域
WELDING_REGION2 = np.array([[650, 220], [770, 1005], [1462, 720], [1420, 181]], np.int32)#放置焊件的识别区域

##焊接穿戴##
SERVER_PORT_WELDING_K1 = '5001'  # 焊接k2服务器端口

SAVE_IMG_PATH_WELDING_K1 = 'images/welding/k1'  # 焊接图片保存在服务器的实际位置

URL_IMG_PATH_WELDING_K1 = f'http://{SERVER_IP}:{SERVER_PORT_WELDING_K1}/{SAVE_IMG_PATH_WELDING_K1}'  # 通过端口映射能够访问的位置 焊接考核科目2

RTSP_WELDING_WEARING='rtsp://admin:yaoan1234@192.168.10.221/cam/realmonitor?channel=1&subtype=0'#焊接的穿戴
WEIGHTS_WELDING_WEARING_HELMET='weights/welding_wearing/welding_wearing_1203.pt'
WEIGHTS_WELDING_WEARING_HUMAN='weights/welding_wearing/yolo11n.pt'

WEIGHTS_WELDING_WEARING=[
    WEIGHTS_WELDING_WEARING_HUMAN,
    WEIGHTS_WELDING_WEARING_HELMET
    
]

###吊篮###
RTSP_BASKET_TOP='rtsp://admin:yaoan1234@192.168.10.213/cam/realmonitor?channel=1&subtype=0'#吊篮顶部视角
RTSP_BASKET_FORNT='rtsp://admin:yaoan1234@192.168.10.215/cam/realmonitor?channel=1&subtype=0'#吊篮正面视角
RTSP_BASKET_SUSPENSION='rtsp://admin:yaoan1234@192.168.10.217/cam/realmonitor?channel=1&subtype=0'#吊篮悬挂机构视角


WEIGHTS_BASKET_SAFETY_BELT_DETECT='weights/basket/1127/basket_safety_belt_1127.pt'
WEIGHTS_BASKET_SEG='weights/basket/1127/basket_seg_1127.pt'
WEIGHTS_BASKET_PERSON_DETECT1='weights/basket/1127/yolo11m-pose1.pt'
WEIGHTS_BASKET_PERSON_DETECT2='weights/basket/1127/yolo11m-pose2.pt'
WEIGHTS_BASKET_WARNING_ZONE_DETECT='weights/basket/1127/basket_warning_zone_1127.pt'


VIDEOS_BASKET=[
    RTSP_BASKET_FORNT,
    RTSP_BASKET_SUSPENSION,
    RTSP_BASKET_TOP
]


WEIGHTS_BASKET=[
    WEIGHTS_BASKET_WARNING_ZONE_DETECT,
    WEIGHTS_BASKET_PERSON_DETECT1,
    WEIGHTS_BASKET_SAFETY_BELT_DETECT,
    WEIGHTS_BASKET_SEG,
    WEIGHTS_BASKET_PERSON_DETECT2    
]

#TODO: 区域里面用元组，可以不用reshape

# BASKET_STEEL_WIRE_REGION = np.array([
#     [(374, 846), (601, 970), (630, 900), (441, 786)],  
#     [(1518, 736), (1649, 945), (2005, 917), (1888, 677)] 
# ], np.int32)


#悬挂机构区域，分为四个区域 D4
BASKET_SUSPENSION_REGION = np.array([
    [[345, 1285], [345, 815], [470, 815], [470, 1285]],
    [[580, 1420], [580, 830], [810, 830], [810, 1420]],
    [[1340, 1320], [1340, 765], [1520, 765], [1520, 1320]],
    [[1915, 1200], [1915, 720], [2100, 720], [2100, 1200]]
], np.int32)


#吊篮上的安全锁
BASKET_SAFETY_LOCK_REGION = np.array([
[[924, 743], [818, 655], [979, 565], [1041, 663]],
[[1780, 613], [1736, 483], [1950, 466], [1949, 591]]
], np.int32)

#吊篮钢丝绳区域
BASKET_STEEL_WIRE_REGION = np.array([
[(780, 502), (989, 580), (859, 704), (691, 642)], # 左一多边形区域
[(1745, 501), (1737, 404), (1950, 391), (1950, 480)] # 右二多边形区域
], np.int32)

#吊篮刷子清洁操作区域
BASKET_CLEANING_OPERATION_REGION = np.array([[(30, 1430), (30, 700), (1900, 550), (1900, 1430)]], np.int32)

#吊篮警戒区域顶部视角
BASKET_WARNING_ZONE_REGION_TOP=np.array([[(1210, 0), (1980, 0), (1980, 400), (1210, 400)]], np.int32)

#吊篮警戒区域正面视角
BASKET_WARNING_ZONE_REGION_FORNT=np.array([[(875, 1440), (875, 770), (1720, 770), (1720, 1440)]], np.int32)

#吊篮空载区域
BASKET_EMPTY_LOAD_REGION = [880, 730, 980, 840]#左上角和右下角

SERVER_PORT_BASKET_K2 = '5005'  # 吊篮k2服务器端口

SAVE_IMG_PATH_BASKET_K2 = 'images/basket/k2'  # 焊接图片保存在服务器的实际位置

URL_IMG_PATH_BASKET_K2 = f'http://{SERVER_IP}:{SERVER_PORT_BASKET_K2}/{SAVE_IMG_PATH_BASKET_K2}'  

##########吊具############
RTSP_EQUIPMENT_TOP='rtsp://admin:yaoan1234@192.168.10.216/cam/realmonitor?channel=1&subtype=0'#吊篮顶部视角
RTSP_EQUIPMENT_FORNT='rtsp://admin:yaoan1234@192.168.10.219/cam/realmonitor?channel=1&subtype=0'#吊篮正面视角

VIDEOS_EQUIPMENT=[
    RTSP_EQUIPMENT_TOP,
    RTSP_EQUIPMENT_FORNT
]

WEIGHTS_EQUIPMENT_SAFETY_BELT_DETECT='weights/equipment/1127/equipment_safety_belt_1127.pt'
WEIGHTS_EQUIPMENT_PERSON_POSE_DETECT='weights/equipment/1127/yolo11m-pose.pt'
WEIGHTS_EQUIPMENT_PERSON_DETECT='weights/equipment/1127/yolo11n.pt'
WEIGHTS_EQUIPMENT_SEAT_PLATE_DETECT='weights/equipment/1127/equipment_seat_plate.pt'

WEIGHTS_EQUIPMENT=[
    WEIGHTS_EQUIPMENT_SAFETY_BELT_DETECT,
    WEIGHTS_EQUIPMENT_SEAT_PLATE_DETECT,
    WEIGHTS_EQUIPMENT_PERSON_POSE_DETECT,
    WEIGHTS_EQUIPMENT_PERSON_DETECT
    
]

SERVER_PORT_EQUIPMENT_K2 = '5006'  # 吊篮k2服务器端口

SAVE_IMG_PATH_EQUIPMENT_K2 = 'images/equipment/k2'  # 焊接图片保存在服务器的实际位置

URL_IMG_PATH_EQUIPMENT_K2 = f'http://{SERVER_IP}:{SERVER_PORT_EQUIPMENT_K2}/{SAVE_IMG_PATH_EQUIPMENT_K2}'

#吊具警戒区域顶部视角
##
EQUIPMENT_WARNING_ZONE_REGION = np.array([
[[668, 375], [750, 0], [1620, 0], [1666, 529]],
], np.int32)
#吊具挂点装置 正面视角
EQUIPMENT_ANCHOR_DEVICE_REGION = np.array([
[[1020, 530], [1020, 20], [1370, 20], [1370, 530]],
], np.int32)

#吊具坐入座板区域，正面视角，在楼梯区域检测到座板
EQUIPMENT_SEATING_PLATE_REGION = np.array([
[[1380, 1385], [1410, 360], [1762, 380], [1695, 1375]],
], np.int32)

#吊具工作绳区域，顶部视角
EQUIPMENT_WORK_ROPE_REGION = np.array([
[[945, 1280], [766, 1109], [1280, 1120], [1780, 1300]],
], np.int32)
#吊具安全绳区域，顶部视角
EQUIPMENT_SAFETY_ROPE_REGION = np.array([
[[945, 1280], [766, 1109], [1280, 1120], [1780, 1300]],
], np.int32)

#吊具清洗操作区域，顶部视角
EQUIPMENT_CLEANING_OPERATION_REGION=np.array([
[[1320, 1350], [1130, 1150], [1930, 1100], [2260, 1260]],
], np.int32)