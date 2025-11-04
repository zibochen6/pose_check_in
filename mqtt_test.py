import paho.mqtt.client as mqtt
import base64
import json

BROKER = "broker.hivemq.com"
PORT = 1883

TOPIC_CMD = "command/send_image"
TOPIC_RESULT = "result/image_data"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Jetson Connected to MQTT Broker")
        client.subscribe(TOPIC_CMD, qos=1)
        print(f"✅ Jetson Subscribed to {TOPIC_CMD}")
    else:
        print(f"❌ Jetson Failed to connect, rc={rc}")

def on_message(client, userdata, msg):
    print(f"📩 Jetson Received MQTT Message on {msg.topic}")
    print(f"📩 Payload Raw: {msg.payload}")

    filename = msg.payload.decode(errors="ignore").strip()
    print(f"📥 Jetson decoded filename = [{filename}]")

    # 检查文件是否存在
    import os
    if not os.path.exists(filename):
        print(f"❌ File does NOT exist: {filename}")
        return

    try:
        with open(filename, "rb") as f:
            img_bytes = f.read()

        img_b64 = base64.b64encode(img_bytes).decode()

        payload = json.dumps({
            "filename": filename,
            "data": img_b64
        })

        client.publish(TOPIC_RESULT, payload, qos=1)
        print(f"✅ Image sent: {filename}")

    except Exception as e:
        print("❌ Error while reading/sending image:", e)


def on_subscribe(client, userdata, mid, granted_qos):
    print(f"✅ Jetson ON_SUBSCRIBE → mid:{mid}, qos:{granted_qos}")

def on_disconnect(client, userdata, rc):
    print("⚠️ Jetson disconnected, rc =", rc)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.on_subscribe = on_subscribe
client.on_disconnect = on_disconnect

print("🔌 Jetson connecting to MQTT broker...")
client.connect(BROKER, PORT, 60)

client.loop_forever()
