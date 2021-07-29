import os

from PIL import Image as PILImage

from tasks.utility import *
from utility_functions.face_utilities import *
from utility_functions.cam2screen import screen2cam, isSupportedDevice


def InitializeROIDetection(inputPath, outputPath, landmarksAlgorithm):
    global input, output, landmarks

    input = inputPath
    output = outputPath
    landmarks = landmarksAlgorithm

# input path is to the folder containing data from various devices
# V1: e.g. /data/Source/EyeCapture/200407/
# V2: e.g. Surface_Pro_4/someuser/00000
def getCaptureSessionDirList(path):
    session_paths = []
    devices = os.listdir(path)
    for device in devices:
        # TODO: Probably the following check is never needed
        if not os.path.isdir(os.path.join(path, device)):
            continue
        users = os.listdir(os.path.join(path, device))
        for user in users:
            sessions = sorted(os.listdir(os.path.join(path, device, user)), key=str)
            for session in sessions:
                session_paths.append(os.path.join(device, user, session))
    return session_paths


def ROIDetectionNewTask(directory, directory_idx, progressbar):
    captures = sorted(getCaptureSessionFileList(os.path.join(input, directory)), key=str)
    total_captures = len(captures)
    # print(directory, total_captures)
    # For directories with no frames folder exit
    if total_captures == 0:
        return

    info_file = os.path.join(input, directory, "info.json")
    screen_file = os.path.join(input, directory, "info.json")


    if os.path.isfile(info_file):
        info_data = json_read(info_file)
    else:
        info_data = {"DeviceName" : "Missing Device Name"}
        print("Error: The file %s doesn't exist. Please investigate."%(info_file))

    if not isSupportedDevice(info_data["DeviceName"]):
            # If the device is not supported in device_metrics_sku.json skip it
            print('%s, %s, Unsupported SKU: %s'%(directory_idx, directory, info_data["DeviceName"]))
            progressbar.addSubProcess(index=directory_idx, max_value=0)
            return

    screen_data = json_read(screen_file)

    # dotinfo.json - { "DotNum": [ 0, 0, ... ],
    #                  "XPts": [ 160, 160, ... ],
    #                  "YPts": [ 284, 284, ... ],
    #                  "XCam": [ 1.064, 1.064, ... ],
    #                  "YCam": [ -6.0055, -6.0055, ... ],
    #                  "Confidence": [ 59.3, 94.2, ... ],
    #                  "Time": [ 0.205642, 0.288975, ... ] }
    #
    # PositionIndex == DotNum
    # Timestamp == Time, but no guarantee on order. Unclear if that is an issue or not
    dotinfo = {
        "DotNum": [],
        "XPts": [],
        "YPts": [],
        "XCam": [],
        "YCam": [],
        "Confidence": [],
        "Time": []
    }

    output_path = os.path.join(output, directory)
    output_frame_path = os.path.join(output_path, "frames")

    faceInfoDict = newFaceInfoDict()

    # frames.json - ["00000.jpg","00001.jpg"]
    frames = []

    facegrid = {
        "X": [],
        "Y": [],
        "W": [],
        "H": [],
        "IsValid": []
    }

    # TODO: Move the split logic out of ROIDetection and into a separate utility module
    if directory_idx % 10 < 8:
        dataset_split = "train"
    elif directory_idx % 10 < 9:
        dataset_split = "val"
    else:
        dataset_split = "test"

    # info.json - {"TotalFrames":99,"NumFaceDetections":97,"NumEyeDetections":56,"Dataset":"train","DeviceName":"iPhone 6"}
    info = {
        "TotalFrames": total_captures,
        "NumFaceDetections": 0,
        "NumEyeDetections": 0,
        "Dataset": dataset_split,
        "DeviceName": info_data["DeviceName"]
    }

    # screen.json - { "H": [ 568, 568, ... ], "W": [ 320, 320, ... ], "Orientation": [ 1, 1, ... ] }
    screen = {
        "H": [],
        "W": [],
        "Orientation": []
    }

    # ensure the output directories exist
    preparePath(output)
    preparePath(output_path)
    preparePath(output_frame_path)

    progressbar.addSubProcess(directory_idx, len(captures))
    for capture_idx, capture in enumerate(captures):
        progressbar.update(directory_idx, capture_idx+1)

        capture_json_path = os.path.join(input, directory, "frames", capture + ".json")
        capture_jpg_path = os.path.join(input, directory, "frames", capture + ".jpg")
        # Returns None if json is corrupt or missing
        capture_data = json_read(capture_json_path)
        # Returns None if image is corrupt or missing
        capture_image = image_read(capture_jpg_path)

        try:
            if capture_data and capture_image:
                capture_image = PILImage.open(capture_jpg_path)
                capture_image_np = np.array(capture_image)  # dlib wants images in numpy array format

                shape_np, isValid = find_face_dlib(capture_image_np)
                info["NumFaceDetections"] = info["NumFaceDetections"] + 1
                if landmarks == "rc":
                    face_rect, left_eye_rect, right_eye_rect, isValid = rc_landmarksToRects(shape_np, isValid)
                    faceInfoDict, faceInfoIdx = rc_faceEyeRectsToFaceInfoDict(faceInfoDict, face_rect, left_eye_rect,
                                                                            right_eye_rect, isValid)
                else:
                    face_rect, left_eye_rect, right_eye_rect, isValid = landmarksToRects(shape_np, isValid)
                    faceInfoDict, faceInfoIdx = faceEyeRectsToFaceInfoDict(faceInfoDict, face_rect, left_eye_rect,
                                                                            right_eye_rect, isValid)

                # facegrid.json - { "X": [ 6, 6, ... ], "Y": [ 10, 10, ... ], "W": [ 13, 13, ... ], "H": [ 13, 13, ... ], "IsValid": [ 1, 1, ... ] }
                if isValid:
                    faceGridX, faceGridY, faceGridW, faceGridH = generate_face_grid_rect(face_rect, capture_image.width,
                                                                                            capture_image.height)
                else:
                    faceGridX = 0
                    faceGridY = 0
                    faceGridW = 0
                    faceGridH = 0

                facegrid["X"].append(faceGridX)
                facegrid["Y"].append(faceGridY)
                facegrid["W"].append(faceGridW)
                facegrid["H"].append(faceGridH)
                facegrid["IsValid"].append(isValid)


                info["NumEyeDetections"] = info["NumEyeDetections"] + 1
                # screen.json - { "H": [ 568, 568, ... ], "W": [ 320, 320, ... ], "Orientation": [ 1, 1, ... ] }
                screen["H"].append(screen_data['H'][capture_idx])
                screen["W"].append(screen_data['W'][capture_idx])
                screen["Orientation"].append(screen_data['Orientation'][capture_idx])

                # dotinfo.json - { "DotNum": [ 0, 0, ... ],
                #                  "XPts": [ 160, 160, ... ],
                #                  "YPts": [ 284, 284, ... ],
                #                  "XCam": [ 1.064, 1.064, ... ],
                #                  "YCam": [ -6.0055, -6.0055, ... ],
                #                  "Confidence": [ 59.3, 94.2, ... ],
                #                  "Time": [ 0.205642, 0.288975, ... ] }
                #
                # PositionIndex == DotNum
                # Timestamp == Time, but no guarantee on order. Unclear if that is an issue or not
                x_raw = capture_data["XRaw"]
                y_raw = capture_data["YRaw"]
                x_cam, y_cam = screen2cam(x_raw,  # xScreenInPoints
                                        y_raw,  # yScreenInPoints
                                        screen_data['Orientation'][capture_idx],  # orientation,
                                        screen_data["W"][capture_idx],  # widthScreenInPoints
                                        screen_data["H"][capture_idx],  # heightScreenInPoints
                                        info_data["DeviceName"])
                confidence = capture_data["Confidence"]

                dotinfo["DotNum"].append(capture_idx)
                dotinfo["XPts"].append(x_raw)
                dotinfo["YPts"].append(y_raw)
                dotinfo["XCam"].append(x_cam)
                dotinfo["YCam"].append(y_cam)
                dotinfo["Confidence"].append(confidence)
                dotinfo["Time"].append(0)  # TODO replace with timestamp as needed

                frame_name = str(f"{capture}.jpg")
                frames.append(frame_name)

                shutil.copyfile(capture_jpg_path, os.path.join(output_frame_path, frame_name))

        except Exception as e:
            print(f"{directory}/frames/{capture}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, e, fname, exc_tb.tb_lineno)

    # write json files
    json_write(os.path.join(output_path, 'frames.json'), frames)
    json_write(os.path.join(output_path, 'screen.json'), screen)
    json_write(os.path.join(output_path, 'info.json'), info)
    json_write(os.path.join(output_path, 'dotInfo.json'), dotinfo)
    json_write(os.path.join(output_path, 'faceGrid.json'), facegrid)
    # write the Face, LeftEye and RightEye
    json_write(os.path.join(output_path, 'dlibFace.json'), faceInfoDict["Face"])
    json_write(os.path.join(output_path, 'dlibLeftEye.json'), faceInfoDict["LeftEye"])
    json_write(os.path.join(output_path, 'dlibRightEye.json'), faceInfoDict["RightEye"])
    return

def getCaptureSessionFileList(path):
    # Check whether it is a valid data/frame directory or not
    if not os.path.isdir(os.path.join(path, "frames")):
        return list()

#    if args.data_format == 'V1':
#        files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(path, "frames")) if f.endswith('.json') and not f == "session.json"]
#    else:
    files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(path, "frames")) if f.endswith('.jpg')]
    return files