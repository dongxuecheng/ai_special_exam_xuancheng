#1 start_detection #开启检测服务

#2 stop_detection #停止检测服务

#3 reset_status #获取复位检测状态

#4 start_exam #开始焊接考核

#5 exam_status #获取焊接考核状态

#6 stop_exam #停止焊接考核

import cv2
from ultralytics import YOLO
from datetime import datetime
import re
from fastapi import FastAPI
import uvicorn
import logging
from fastapi.staticfiles import StaticFiles
import multiprocessing as mp
from multiprocessing import Queue
from utils import IoU_polygon,IoU
from config import SAVE_IMG_PATH_WELDING_K2,URL_IMG_PATH_WELDING_K2,WEIGHTS_WELDING,VIDEOS_WELDING,WELDING_REGION1,WELDING_REGION2

#焊接考核的穿戴
app = FastAPI()
# 挂载目录作为静态文件路径
app.mount("/images", StaticFiles(directory="images"))
# 获得uvicorn服务器的日志记录器
logging = logging.getLogger("uvicorn")

# 全局变量
processes = []
start_events = []  # 存储每个进程的启动事件
stop_events = []  # 存储每个进程的停止事件

#mp.Array性能较高，适合大量写入的场景
welding_reset_flag = mp.Array('b', [False] * 5) # 创建一个长度为5的共享数组，并初始化为False,用于在多个线程中间传递变量
welding_exam_flag = mp.Array('b', [False] * 14)  # 创建一个长度为5的共享数组，并初始化为False,用于在多个线程中间传递变量
#mp.Value适合单个值的场景，性能较慢
manager = mp.Manager()
welding_reset_imgs = manager.dict()  #用于存储各个步骤的图片
welding_exam_imgs = manager.dict()  #用于存储焊接考核各个步骤的图片
welding_exam_order = manager.list()#用于存储焊接考核各个步骤的顺序

exam_status_flag = mp.Value('b', False)  # 创建一个共享变量，并初始化为False,用于在多个线程中间传递变量,表示是否开始考核,True表示开始考核
frame_queue_list = [Queue(maxsize=50) for _ in range(5)]  # 创建5个队列，用于存储视频帧


def fetch_video_stream(rtsp_url, frame_queue_list, start_event, stop_event):  # 拉取视频流到队列中
    cap = cv2.VideoCapture(rtsp_url)
    index = VIDEOS_WELDING.index(rtsp_url)
    while cap.isOpened():
        if stop_event.is_set():  # 控制停止推理
            logging.info("fetch_video_stream is stopped")
            break
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 8 != 0:
            continue
        if not start_event.is_set():
            start_event.set()
            logging.info(f'fetch_video_stream{rtsp_url}')
        if index == 3:#焊台视频流需要放入两个队列
            frame_queue_list[index+1].put_nowait(frame)
        frame_queue_list[index].put_nowait(frame)
    cap.release()

def infer_yolo(model_path, video_source, start_event, stop_event,welding_reset_flag, welding_reset_imgs,welding_exam_flag, welding_exam_imgs,welding_exam_order):#YOLO模型推理
    model = YOLO(model_path)
    while True:      
        if stop_event.is_set():
            logging.info(f"{model_path} infer_yolo is stopped")
            break        
        if video_source.empty():
            continue
        frame = video_source.get()
        #results = model.track(frame,verbose=False,conf=0.5,device='0',tracker="bytetrack.yaml")
        if model_path==WEIGHTS_WELDING[4]:
            results = model.predict(frame,verbose=False,conf=0.4)
        else:
            results = model.track(frame,verbose=False,conf=0.5,tracker="bytetrack.yaml",persist=True)
        if not start_event.is_set():
            start_event.set()
            logging.info(f"{model_path} infer_yolo is running")
        if model_path==WEIGHTS_WELDING[4]:
            if results[0].probs.top1conf>0.8:
                label=model.names[results[0].probs.top1]
                if label=='welding':
                    welding_exam_flag[6]=True
        else:            
            boxes = results[0].boxes.xyxy  
            confidences = results[0].boxes.conf  
            classes = results[0].boxes.cls  

            if model_path==WEIGHTS_WELDING[2]:
                welding_exam_flag[0]=True#油桶每次都要检测，初始化为True
                welding_reset_flag[0] = True

            if model_path==WEIGHTS_WELDING[3]:
                grounding_wire_flag=False
                welding_components_flag=False

            if model_path==WEIGHTS_WELDING[3]:
                #实时获取焊件和锤子的位置
                welding_components_box=[0,0,0,0]
                hamer_box=[0,0,0,0]

            for i in range(len(boxes)):
                cls = int(classes[i].item())
                label = model.names[cls]
                if model_path==WEIGHTS_WELDING[0]:
                    if label=="broom":
                        welding_exam_flag[13]=True

                if model_path==WEIGHTS_WELDING[1]:
                    if label== "machine_open":#检测焊机开关
                        welding_exam_flag[4] = True
                        welding_reset_flag[4] = True

                    if label=="machine_close" and welding_exam_flag[4]:#当打开过焊机开关，才能检测关闭状态
                        welding_exam_flag[8] = True
                        welding_reset_flag[4] = False

                if model_path==WEIGHTS_WELDING[2]:
                    if label=="oil_tank":#检测油桶
                        welding_exam_flag[0]=False
                        welding_reset_flag[0]=False
                    if label=="gloves":
                        welding_exam_flag[7]=True
                    if label=="main_switch_open":
                        welding_exam_flag[1]=True
                        welding_reset_flag[1]=True
                    if label=="main_switch_close":
                        welding_reset_flag[1]=False
                        if welding_exam_flag[1]:
                            welding_exam_flag[12]=True


                if model_path==WEIGHTS_WELDING[3]:
                    if label=="grounding_wire":
                        if IoU_polygon(boxes[i].tolist(), polygon_points=WELDING_REGION1.tolist())>0:
                            grounding_wire_flag=True
                            welding_reset_flag[2]=True
                            welding_exam_flag[2]=True#连接搭铁线
                        else:
                            welding_reset_flag[2]=False
                            if welding_exam_flag[2]:
                                #logging.info("检测到搭铁线，但是不在桌上")
                                welding_exam_flag[9]=True
                            
                    if label=="welding_component":
                        if IoU_polygon(boxes[i].tolist(), polygon_points=WELDING_REGION2.tolist())>0:
                            welding_components_flag=True
                            welding_exam_flag[3]=True
                            welding_reset_flag[3]=True
                            welding_components_box=boxes[i].tolist()
                        else:
                            if welding_exam_flag[3]:
                                #logging.info("检测到焊件，但是不在桌上")
                                welding_exam_flag[10]=True

                    if label=="mask":
                        welding_exam_flag[5]=True
                    if label=="hamer":
                        hamer_box=boxes[i].tolist()
                        if IoU(hamer_box,welding_components_box)>0:
                            welding_exam_flag[11]=True#拿起锤子打扫焊件

                    #TODO：若图片中没有检测到搭铁线或焊件，则代表已经取下
                    if not grounding_wire_flag and welding_exam_flag[2] and welding_exam_flag[6]:
                        welding_exam_flag[9]=True
                        #logging.info("没有检测到搭铁线")
                    if not welding_components_flag and welding_exam_flag[3] and welding_exam_flag[6]:
                        #logging.info("没有检测到焊件")
                        welding_exam_flag[10]=True


        reset_steps = {
            WEIGHTS_WELDING[1]: [(welding_reset_flag[4], 'reset_step_5')],
            WEIGHTS_WELDING[2]: [
            (welding_reset_flag[1], 'reset_step_2'),
            (welding_reset_flag[0], 'reset_step_1')
            ],
            WEIGHTS_WELDING[3]: [
            (welding_reset_flag[2], 'reset_step_3'),
            (welding_reset_flag[3], 'reset_step_4')
            ]
        }

        if not exam_status_flag.value and model_path in reset_steps:
            for flag, step in reset_steps[model_path]:
                if flag and step not in welding_reset_imgs:
                    logging.info(f"{step} is not reset")
                    save_image_reset(welding_reset_imgs, results, step)




        exam_steps = {
                WEIGHTS_WELDING[0]: [
                    (welding_exam_flag[13], "welding_exam_14")
                ],
                WEIGHTS_WELDING[1]: [
                    (welding_exam_flag[4], "welding_exam_5"),
                    (welding_exam_flag[8], "welding_exam_9")
                ],
                WEIGHTS_WELDING[2]: [
                    (welding_exam_flag[0], "welding_exam_1"),
                    (welding_exam_flag[1], "welding_exam_2"),
                    (welding_exam_flag[7], "welding_exam_8"),
                    (welding_exam_flag[12], "welding_exam_13")
                ],
                WEIGHTS_WELDING[3]: [
                    (welding_exam_flag[2], "welding_exam_3"),
                    (welding_exam_flag[9], "welding_exam_10"),
                    (welding_exam_flag[3], "welding_exam_4"),
                    (welding_exam_flag[5], "welding_exam_6"),
                    (welding_exam_flag[10], "welding_exam_11"),
                    (welding_exam_flag[11], "welding_exam_12")
                ],
                WEIGHTS_WELDING[4]: [
                    (welding_exam_flag[6], "welding_exam_7")
                ]
            }

        if exam_status_flag.value and model_path in exam_steps:
            for flag, step in exam_steps[model_path]:
                if flag and step not in welding_exam_imgs:
                    save_image_exam(welding_exam_imgs, results, step, welding_exam_order)




def save_image_reset(welding_reset_imgs,results, step_name):#保存图片
    save_time = datetime.now().strftime('%Y%m%d_%H%M')
    img_path = f"{SAVE_IMG_PATH_WELDING_K2}/{step_name}_{save_time}.jpg"
    url_path = f"{URL_IMG_PATH_WELDING_K2}/{step_name}_{save_time}.jpg"
    annotated_frame = results[0].plot()

    cv2.imwrite(img_path, annotated_frame)
    welding_reset_imgs[step_name]=url_path

def save_image_exam(welding_exam_imgs,results, step_name,welding_exam_order):
    save_time = datetime.now().strftime('%Y%m%d_%H%M')
    img_path = f"{SAVE_IMG_PATH_WELDING_K2}/{step_name}_{save_time}.jpg"
    url_path = f"{URL_IMG_PATH_WELDING_K2}/{step_name}_{save_time}.jpg"
    annotated_frame = results[0].plot()
    cv2.imwrite(img_path, annotated_frame)
    welding_exam_imgs[step_name]=url_path
    welding_exam_order.append(step_name)
    logging.info(f"{step_name}完成")

def reset_shared_variables():
    global frame_queue_list
    exam_status_flag.value = False
    init_reset_variables()
    init_exam_variables()
    for queue in frame_queue_list:
        while not queue.empty():
            queue.get()
            logging.info("frame_queue_list is empty")

def init_exam_variables():
    for i in range(len(welding_exam_flag)):
        welding_exam_flag[i] = False    
    welding_exam_imgs.clear()
    welding_exam_order[:]=[]


def init_reset_variables():
    for i in range(len(welding_reset_flag)):
        welding_reset_flag[i] = False
    welding_reset_imgs.clear()

@app.get('/start_detection')
def start_detection():#发送开启AI服务时，检测复位
    if not any(p.is_alive() for p in processes):  # 防止重复开启检测服务
        for video_source in VIDEOS_WELDING:
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            stop_event=mp.Event()
            process = mp.Process(target=fetch_video_stream, args=(video_source,frame_queue_list, start_event, stop_event))
            stop_events.append(stop_event)
            start_events.append(start_event)  # 加入 start_events 列表，因为start_events是列表，append或clear不需要加global
            processes.append(process)
            process.start()

        for model_path, video_source in zip(WEIGHTS_WELDING, frame_queue_list):
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            stop_event=mp.Event()
            process = mp.Process(target=infer_yolo, args=(model_path,video_source, start_event, stop_event, welding_reset_flag, welding_reset_imgs,welding_exam_flag, welding_exam_imgs,welding_exam_order))
            start_events.append(start_event)  # 加入 start_events 列表
            stop_events.append(stop_event)
            processes.append(process)
            process.start()
        # 等待所有进程的 start_event 被 set
        for event in start_events:
            event.wait()  # 等待每个进程通知它已经成功启动
            
        logging.info("start_detection is success")
        exam_status_flag.value = False#表示没有开始考核
        return {"status": "SUCCESS"}
    else:
        logging.info("welding_reset_detection——ALREADY_RUNNING")
        return {"status": "ALREADY_RUNNING"}
    
@app.get('/start_exam')
def start_exam():#发送开启AI服务时
    if not exam_status_flag.value:  # 防止重复开启检测服务
        exam_status_flag.value = True
        init_exam_variables()
        logging.info('start_exam')
        return {"status": "SUCCESS"}
    else:
        logging.info("start_exam is already running")
        return {"status": "ALREADY_RUNNING"}

@app.get('/reset_status')#获取复位检测状态
def reset_status():
    if not any(welding_reset_flag):  # 表明不需要复位,如果 welding_reset_flag 列表中的所有元素都为 False，则 any(welding_reset_flag) 返回 False，not any(welding_reset_flag) 返回 True。
        logging.info('reset_all!')
        return {"status": "RESET_ALL"}
    else:
        logging.info('reset_all is false')
        json_array = [
            {"resetStep": re.search(r'reset_step_(\d+)', key).group(1), "image": value}
            for key, value in welding_reset_imgs.items()
        ]
        init_reset_variables()#初始化复位变量
        return {"status": "NOT_RESET_ALL", "data": json_array}

            
@app.get('/exam_status')
def exam_status():
    if not welding_exam_order:#使用not来判断列表是否为空
        logging.info('welding_exam_order is none')
        return {"status": "NONE"}
    else:
        json_array = [
            {"step": re.search(r'welding_exam_(\d+)', value).group(1), "image": welding_exam_imgs.get(value)}
            for value in welding_exam_order
        ]
        return {"status": "SUCCESS", "data": json_array}

@app.get('/stop_exam')
def stop_exam():
    if exam_status_flag.value:
        exam_status_flag.value = False
        logging.info('stop_exam')
        return {"status": "SUCCESS"}
    
    
#停止多进程函数的写法
def stop_inference_internal():
    if processes:  # 检查是否有子进程正在运行
            # 停止所有进程
        for stop_event in stop_events:
            stop_event.set()
        for process in processes:
            if process.is_alive():
                #打印当前进程的pid
                process.join(timeout=1)  # 等待1秒
                if process.is_alive():
                    logging.warning('Process did not terminate, forcing termination')
                    process.terminate()  # 强制终止子进程
                
        processes.clear()  # 清空进程列表，释放资源
        start_events.clear()
        stop_events.clear()
        logging.info('detection stopped')
        return True
    else:
        logging.info('No inference stopped')
        return False

@app.get('/stop_detection')
def stop_detection():
    if stop_inference_internal():      
        reset_shared_variables()  
        return {"status": "DETECTION_STOPPED"}
    else:
        return {"status": "No_detection_running"}

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.10.109", port=5002)
    #uvicorn.run(app, host="127.0.0.1", port=5002)
