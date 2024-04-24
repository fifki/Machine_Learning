import math
import boto3
import cv2
import torch
import json #to save the propmt
import base64 # to decode the image 
import time
import numpy as np


from PIL import Image #to print the image
from torchvision.models.segmentation import deeplabv3_resnet50
from ultralytics import YOLO, SAM # YOLO: You Only Look Once - SAM Segment Anything Model    deeplabv3_mobilenet_v3_large
from torchvision import transforms
from flask import Flask, render_template, request, Response


app = Flask(__name__)



def stream_object_detection():
	# Initialize video stream
	video_stream = cv2.VideoCapture(0)
	video_stream.set(3, 640)  # Set video width
	video_stream.set(4, 480)  # Set video height

	# Load YOLO model
	model = YOLO("yolo-Weights/yolov8n.pt")  # Ensure your model weights path is correct

	# Define class names (assuming these are the classes your YOLO model recognizes)
	classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
			    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
			    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
				"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
				"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
				"fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
				"carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
				"diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
				"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
				"teddy bear", "hair drier", "toothbrush"
				]

	while True:
		success, img = video_stream.read()
		if not success:
			break

		# Run object detection
		results = model(img, stream=True)

		# Process results
		for r in results:
			boxes = r.boxes
			for box in boxes:
				x1, y1, x2, y2 = map(int, box.xyxy[0])
				confidence = math.ceil((box.conf[0] * 100)) / 100
				cls = int(box.cls[0])
				cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
				cv2.putText(img, f"{classNames[cls]} {confidence:.2f}", (x1, y1 - 10),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

				# Encode the frame in JPEG format
				_, buffer = cv2.imencode('.jpg', img)
				frame = buffer.tobytes()

				yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')			


def stream_segmentation():
	# Initialize video stream
	video_stream = cv2.VideoCapture(0)
	video_stream.set(3, 640)  # Set video width
	video_stream.set(4, 480)  # Set video height

	# Load deeplabv3_mobilenet_v3_large
	model = deeplabv3_resnet50(pretrained=True)  # Ensure your model weights path is correct
	model.eval()

	while True:
		success, frame = video_stream.read()
		if not success:
			break

		# Preprocess the image
		preprocess = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((256, 256)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		input_tensor = preprocess(frame)
		input_batch = input_tensor.unsqueeze(0)

		with torch.no_grad():
			output = model(input_batch)['out'][0]
		output_predictions = output.argmax(0)

		# Create a color mask
		r = np.zeros_like(output_predictions).astype(np.uint8)
		g = np.zeros_like(output_predictions).astype(np.uint8)
		b = np.zeros_like(output_predictions).astype(np.uint8)
		r[output_predictions == 15] = 255  # Class 15 is people, change this based on your needs
		segmented_output = np.stack([r, g, b], axis=2)

		# Resize to match the original image
		segmented_output = cv2.resize(segmented_output, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

		# Encode the frame in JPEG format
		_, buffer = cv2.imencode('.jpg', segmented_output)
		frame = buffer.tobytes()

		yield (b'--frame\r\n'
		b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def depth_estimation_prediction(image_path):
	
	filename = image_path
	model_type = "MiDaS_small"     

	midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
	midas.eval() #evaluation mode


	midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
	transform = midas_transforms.small_transform


	img = cv2.imread("static/" + filename)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	input_batch = transform(img)

	with torch.no_grad():
		prediction = midas(input_batch)

		prediction = torch.nn.functional.interpolate(
												prediction.unsqueeze(1),
												size=img.shape[:2],
												mode="bicubic",
												align_corners=False,
												).squeeze()

	output = prediction.numpy() #its of type : n-D numpy array means matrix
	output = Image.fromarray(output) # Convert the numpy matrix to an image (to be saved)

	if output.mode != "RGB":
		output = output.convert("RGB")

	depth_image_path = f"depth_image_{time.strftime('%Y%m%d_%H%M%S')}.png"
	output.save("static/" + depth_image_path)

	return depth_image_path

def generate_image(prompt):

    bedrock_runtime = boto3.client(
        aws_access_key_id="Your_access_key",
        aws_secret_access_key="your_secret_key",
        region_name="us-east-1",
        service_name='bedrock-runtime'
    )

    request_body = json.dumps({
                "text_prompts": [
                    {
                    "text": prompt
                    }
                ],
                "cfg_scale": 10,  # Very low cfg_scale
                "seed": 0,  # Use a seed for reproducability
                "samples" : 1,
                })
    
    response = bedrock_runtime.invoke_model(body=request_body, modelId="stability.stable-diffusion-xl-v1")
    response_body = json.loads(response.get('body').read())
    
    base64_image_data = base64.b64decode(response_body["artifacts"][0]["base64"])
    
    # Images/image_20120515_155045.png
    file_path = f"image_{time.strftime('%Y%m%d_%H%M%S')}.png"
    with open("static/" + file_path, "wb") as file:
        file.write(base64_image_data)

    return file_path




@app.route('/video_feed')
def video_feed():

    return Response(stream_object_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_segmentation')
def video_feed_segmentation():
 
    return Response(stream_segmentation(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index_page():
   return render_template("index.html")

@app.route('/object_detection')
def object_detection_page():
   return render_template("object_detection.html")


@app.route('/segmentation')
def segmentation_page():
   return render_template("segmentation.html")


@app.route('/depth_estimation', methods=['GET', 'POST'])
def depth_estimation_page():
	if request.method == 'POST':
		image = request.files["image"]
		if image :
			image_path = image.filename
			image.save("static/" + image_path)

		depth_image_path = depth_estimation_prediction(image_path)

		return render_template("depth_estimation.html", image=image_path, depth_image=depth_image_path)
	
	return render_template("depth_estimation.html" , image=None, depth_image=None)


@app.route('/image_generator', methods=['GET', 'POST'])
def image_generator_page():
	if request.method == 'POST':
		prompt = request.form["prompt"]
		image_path = generate_image(prompt)

		return render_template("image_generator.html", prompt=prompt, result=image_path)
	else: 
		return render_template("image_generator.html", prompt=None, result=None)
   #return render_template("image_generator.html")



if __name__ == '__main__':
   app.run(debug=True)	
