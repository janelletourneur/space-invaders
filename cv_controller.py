import websockets
import asyncio
import cv2
import time
import mediapipe as mp # python at most 3.12 , mediapipe==0.10.21
import numpy as np

mp_hands=mp.solutions.hands
hand = mp_hands.Hands()

def dist(marker1,marker2):
    a1=np.array((marker1.x,marker1.y,marker1.z))
    a2=np.array((marker2.x,marker2.y,marker2.z))
    return np.linalg.norm(a1-a2)


def input_to_command(lmdhands):
    ldms=lmdhands[0].landmark
        
    if len(ldms)<21: return None
    index_tip=ldms[8]
    
    index_mcp=ldms[5]
    
    thumb_tip=ldms[4]
    
    thumb_ip=ldms[3]
    
    thumb_mcp=ldms[2]
    
    middle_finger_tip=ldms[12]
    
    middle_finger_mcp=ldms[9]
    
    ring_finger_tip=ldms[16]
    
    ring_finger_mcp=ldms[13]
    
    scale=dist(thumb_ip,thumb_tip)
    alpha=2.6
    if dist(thumb_tip,index_tip) < scale:
        return None
    elif (dist(middle_finger_tip,ring_finger_tip)>(alpha/1.8)*scale) and ((middle_finger_tip.y-middle_finger_mcp.y)<-alpha*scale*1.2) and ((ring_finger_tip.y-ring_finger_mcp.y)<-alpha*scale*1.2):
        return "a"
    elif (index_tip.y-index_mcp.y)>alpha*scale:
        return "ENTER"
    elif (index_tip.y-index_mcp.y)<-alpha*scale:
        return "FIRE"
    elif ((index_tip.x-index_mcp.x)>alpha*scale):
        return "LEFT"
    elif ((index_tip.x-index_mcp.x)<-alpha*scale) or ((thumb_tip.x-thumb_mcp.x)<-alpha*0.5*scale):
        return "RIGHT"
        
    return None
    
    
    




async def send_command():
    uri = "ws://localhost:8765"
    cap= cv2.VideoCapture(1)
    
    async with websockets.connect(uri) as websocket:
        print("Connected! Controls:")
        print("position de repos sans commandes: index et pouce qui se touche pour faire un ok")
        print("pointer vers la gauche avec l'index = LEFT")
        print("pointer vers la droite avec le pouce ou l'index = RIGHT")
        print("pointer vers le haut avec l'index = FIRE")
        print("pointer vers le bas avec l'index = ENTER")
        print("salut vulcain : quitter ")
        
        
        # Get the current event loop
        loop = asyncio.get_running_loop()

        # Main control loop
        while True:
            # Get input from user asynchronously using run_in_executor
            cmd=None
            success,frame =cap.read()
            
            if not success: break

            RGB_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            input=hand.process(RGB_frame)
            
            if input.multi_hand_landmarks:
                cmd=input_to_command(input.multi_hand_landmarks)
            else: continue

            if cmd is not None:
                print(f"Received: {cmd}")
                
            if cmd == "a":
                print("Exiting control module")
                continue
            elif cmd is not None:
                await websocket.send(cmd)
                print(f"Sent: {cmd}")
                time.sleep(0.05)
                


# Run the control module
if __name__ == "__main__":
    asyncio.run(send_command()) 
    
cap.release()