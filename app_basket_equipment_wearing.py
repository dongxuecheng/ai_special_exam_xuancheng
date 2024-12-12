import numpy as np
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

from utils import IoU
from config import SAVE_IMG_PATH_BASKET_EQUIPMENT_K1,URL_IMG_PATH_BASKET_EQUIPMENT_K1,WEIGHTS_BASKET_EQUIPMENT_WEARING,RTSP_BASKET_EQUIPMENT_WEARING
import time


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
wearing_human_in_postion=mp.Value('b', False)  # 用来判断人是否在指定位置
wearing_items_nums=mp.Array('i', [0] * 2)  # 用来存储穿戴物品的数量
wearing_detection_img_flag=mp.Value('b', False)  # 用来传递穿戴检测图片的标志，为真时，表示保存图片
#mp.Value适合单个值的场景，性能较慢
manager = mp.Manager()
wearing_detection_img = manager.dict()  #用于存储检测焊接穿戴图片

frame_queue_list = [Queue(maxsize=50) for _ in range(2)] 



def fetch_video_stream(rtsp_url, frame_queue_list, start_event, stop_event):  # 拉取视频流到队列中
    #队列与对应的模型
    #frame_queue_list[0]:检测人
    #frame_queue_list[1]:检测穿戴
    cap = cv2.VideoCapture(rtsp_url)
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
        frame_queue_list[0].put_nowait(frame)
        frame_queue_list[1].put_nowait(frame)
    cap.release()

def infer_yolo(model_path,video_source, start_event, stop_event,wearing_human_in_postion, wearing_items_nums, wearing_detection_img_flag, wearing_detection_img):#YOLO模型推理
    model = YOLO(model_path)
    while True:      
        if stop_event.is_set():
            logging.info(f"{model_path} infer_yolo is stopped")
            break        
        if video_source.empty():
            continue
        frame = video_source.get()
        #results = model.track(frame,verbose=False,conf=0.5,device='0',tracker="bytetrack.yaml")
        #results = model.predict(frame, verbose=False, conf=0.3)
        if model_path==WEIGHTS_BASKET_EQUIPMENT_WEARING[0]:#yolov8s，专门用来检测人
            #model.classes = [0]#设置只检测人一个类别
            results = model.predict(frame,conf=0.6,verbose=False,classes=[0])#这里的results是一个生成器
            wearing_human_in_postion.value=False
        else:
            results = model.predict(frame,conf=0.7,verbose=False)

        if not start_event.is_set():
            start_event.set()
            logging.info(f"{model_path} infer_yolo is running")


            ##下面这些都是tensor类型
        boxes = results[0].boxes.xyxy  # 提取所有检测到的边界框坐标
        confidences = results[0].boxes.conf  # 提取所有检测到的置信度
        classes = results[0].boxes.cls  # 提取所有检测到的类别索引
        ###劳保,不在函数外部定义是因为需要每一帧重新赋值
        wearing_items={
                'belt': 0,
                'helemt': 0
        }
            
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            confidence = confidences[i].item()
            cls = int(classes[i].item())
            label = model.names[cls]

            
            # if x1 < WEAR_DETECTION_AREA[0] or y1 < WEAR_DETECTION_AREA[1] or x2 > WEAR_DETECTION_AREA[2] or y2 > WEAR_DETECTION_AREA[3]:
            #     continue  # 跳过不在区域内的检测框
            
            #if model_path==WEIGHTS_BASKET_EQUIPMENT_WEARING[0]:#yolov8s，专门用来检测人
            if label=="person" and not wearing_human_in_postion.value:
                if IoU([x1,y1,x2,y2],[876, 0, 1923, 1440]) > 0:
                    wearing_human_in_postion.value=True
                    
            else:
                if IoU([x1,y1,x2,y2],[876, 0, 1923, 1440]) > 0:
                    wearing_items[label] += 1


        if model_path==WEIGHTS_BASKET_EQUIPMENT_WEARING[1]:

            if wearing_human_in_postion.value and not wearing_detection_img_flag.value:
                wearing_items_nums[0] = max(wearing_items_nums[0], wearing_items["belt"])
                wearing_items_nums[1] = max(wearing_items_nums[1], wearing_items["helemt"])

            if wearing_detection_img_flag.value and 'wearing_img' not in wearing_detection_img:
                save_time=datetime.now().strftime('%Y%m%d_%H%M')
                imgp_ath = f"{SAVE_IMG_PATH_BASKET_EQUIPMENT_K1}/wearing_detection_{save_time}.jpg"
                post_path= f"{URL_IMG_PATH_BASKET_EQUIPMENT_K1}/wearing_detection_{save_time}.jpg"
                annotated_frame = results[0].plot()
                cv2.imwrite(imgp_ath, annotated_frame)
                wearing_detection_img['wearing_img']=post_path
            







        


def reset_shared_variables():
    global frame_queue_list
    init_exam_variables()
    for queue in frame_queue_list:
        while not queue.empty():
            queue.get()
            logging.info("frame_queue_list is empty")

def init_exam_variables():
    for i in range(len(wearing_items_nums)):
        wearing_items_nums[i] = False    
    wearing_human_in_postion.value = False
    wearing_detection_img_flag.value = False
    wearing_detection_img.clear()



@app.get('/wearing_detection')
def wearing_detection():
    if not any(p.is_alive() for p in processes):  # 防止重复开启检测服务
        #for video_source in VIDEOS_EQUIPMENT:
        start_event = mp.Event()  # 为每个进程创建一个独立的事件
        stop_event=mp.Event()
        process = mp.Process(target=fetch_video_stream, args=(RTSP_BASKET_EQUIPMENT_WEARING,frame_queue_list, start_event, stop_event))
        stop_events.append(stop_event)
        start_events.append(start_event)  # 加入 start_events 列表，因为start_events是列表，append或clear不需要加global
        processes.append(process)
        process.start()

        for model_path, video_source in zip(WEIGHTS_BASKET_EQUIPMENT_WEARING, frame_queue_list):
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            stop_event=mp.Event()
            process = mp.Process(target=infer_yolo, args=(model_path,video_source, start_event, stop_event,wearing_human_in_postion, wearing_items_nums, wearing_detection_img_flag, wearing_detection_img))
            start_events.append(start_event)  # 加入 start_events 列表
            stop_events.append(stop_event)
            processes.append(process)
            process.start()
        # 等待所有进程的 start_event 被 set
        for event in start_events:
            event.wait()  # 等待每个进程通知它已经成功启动
            
        logging.info("start_detection is success")
        #exam_status_flag.value = False#表示没有开始考核
        return {"status": "SUCCESS"}
    else:
        logging.info("reset_detection——ALREADY_RUNNING")
        return {"status": "ALREADY_RUNNING"}
    
@app.get('/human_postion_status')
def human_postion_status():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测
    if not wearing_human_in_postion.value:
        logging.info('NOT_IN_POSTION')
        #return jsonify({"status": "NOT_IN_POSTION"}), 200
        return {"status": "NOT_IN_POSTION"}
    else:
        logging.info('IN_POSTION')
        #return jsonify({"status": "IN_POSTION"}), 200
        return {"status": "IN_POSTION"}


@app.get('/wearing_status')
def wearing_status():

    wearing_detection_img_flag.value=True
    time.sleep(1)
    if 'wearing_img' not in wearing_detection_img or not wearing_human_in_postion.value:

        return {"status": "NONE"}
    else:

        wearing_items_list = [ 'helmet', 'gloves', 'shoes']
        json_array = []
        for num, item in zip(wearing_items_nums, wearing_items_list):
            json_object = {"name": item, "number": num}
            json_array.append(json_object)

        logging.info(json_array)
        image=wearing_detection_img['wearing_img']
        logging.info(image)

        return {"status": "SUCCESS","data":json_array,"image":image}

@app.get('/end_wearing_exam')
def end_wearing_exam():
    #stop_inference_internal()
    reset_shared_variables()
    #return jsonify({"status": "SUCCESS"}), 200
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
    #uvicorn.run(app, host="192.168.10.109", port=5003)
    uvicorn.run(app, host="127.0.0.1", port=5003)
