# Capture Data Format for GazeCapture pytorch

Schema Version 200407

## Overview

Each capture session creates a new subdirectory for data capture.

Our data captures are stored in a directory structure of:

```
schema version / device sku / user hash / session timestamp in ISO 8601
```

for example

```
200407/Surface_Pro_6_1796_Commercial/g1yfT+gSdqpLwVXmevzNDw/2020-08-12T04:49:38/
```

There are a number of optional metadata files.  Note that for our dataset, DeviceSKU
is part of the directory structure and device metrics can be inferred from the device name.

```
 screen.json: {
  H: ...,
  W: ...,
  Orientation: ...
 }
 
 info.json: {
   DeviceName: ...,
   ReferenceEyeTracker: ...
 }

 deviceMetrics.json {
   xCameraToScreenDisplacementInCm: ...,
   yCameraToScreenDisplacementInCm: ...,
   widthScreenInCm: ...,
   heightScreenInCm: ...,
   ppi: ...
     }
```

Frame data is captured into a subdirectory of the session directory named 'frames' which contains pairs of image and metadata files.
Each frame data pair shares the same root filename, e.g. frameId.  A simple frameid convention is %gazeTargetIndex%-%cameraSnapshotIndex%.
For example 00001-00015.jpg/00001-00015.json, 00003-00007.jpg/00003-00007.json, ...

```
  %frameId%.jpg (close to lossless compression as possible)
  
  %frameId%.json: {
    XRaw: ...,
    YRaw: ...,
    Confidence: ...
  }
```

for example:

```
frames/00032-00019.jpg
frames/00032-00019.json
```

Confidence is set to 'pixel distance of gaze position from gaze target as measured by reference eye tracker'.
This value can be used to filter out data points where the user was not actively looking at the gaze target when the camera snapshot was taken.
It can also be used to make relative accuracy observations between the reference eye tracker and the DeepEyes prediction model results.
If a reference eye tracker was not attached and in use during the capture session, then Confidence is set to -1.

This data is then post-processed using DLib or similar head/eye feature detection to add metadata files:

```
 dotCam.json: {
   XCam: ...,
   YCam: ...
 }
 
 faceGrid: {
   X: ...,
   Y: ...,
   W: ...,
   H: ...,
   IsValid: ...
 }
 
 leftEyeGrid: {
   X: ...,
   Y: ...,
   W: ...,
   H: ...,
   IsValid: ...
 }
 
 rightEyeGrid: {
   X: ...,
   Y: ...,
   W: ...,
   H: ...,
   IsValid: ...
 }
```
And extracted images:

```
 face.jpg
 leftEye.jpg
 rightEye.jpg
```

## Changes to schema

Use DeviceSku in place of DeviceName (e.g. different metrics for different Surface Book 2s with different screen sizes)

Capturing raw pixels vs. scaled pixels
    e.g. dotInfo.json should contain unscaled device pixels on screen (e.g. zoom mode independent)

## Design Summary

Capture -> Prepare -> ML

### Capture

#### This is how MIT GazeCapture stores the data:

```
{dataHome}/{sessionId}/frames/{frameId}.jpg
{dataHome}/{sessionId}/frames.json              JSON array of jpg file names
{dataHome}/{sessionId}/dotInfo.json             JSON array of X/Y target point for each frame
    Arrays: DotNum, XPts, YPts, XCam, YCam, Time
{dataHome}/{sessionId}/info.json                Facial feature recognition metadata &amp; device type
    TotalFrames, NumFaceDetections, NumEyeDetections, Dataset (train/validate/test), DeviceName
{dataHome}/{sessionId}/screen.json              Screen W/H/Orientation for frames
    Arrays: H, W, Orientation
```

#### This is how we will store the data from EyeCapture

```
    /{dataHome}/{schemaVersion}/{deviceSku}/{userNameHash}/{sessionId}/*.json
    /{dataHome}/{schemaVersion}/{deviceSku}/{userNameHash}/{sessionId}/frames/*.json & *.jpg
```

e.g.

```
    /data/200407/Surface_Pro_6_1796_Commercial/P0F_+nViS55W3yNOti3bXw==/2020-07-10T02:22:12/frames/00004-00021.json
    /data/200407/Surface_Pro_6_1796_Commercial/P0F_+nViS55W3yNOti3bXw==/2020-07-10T02:22:12/frames/00004-00021.jpg
```

#### Further Notes (now obsolete?)
```
{frameId}.jpg       Camera Images in JPG Lossless
{frameId}.json      { "XRaw":..., "YRaw":..., "Confidence":... }
frames.json         JSON array of jpg file names
dotInfo.json        JSON arrays: DotNum, XPts, YPts, XCam, YCam, Time
                    XPts/YPts are in device dependent pixel coordinates, unaffected by display zoom.
info.json
                    TotalFrames, NumFaceDetections, NumEyeDetections, Dataset (train/validate/test), DeviceName
screen.json         JSON arrays: H, W, Orientation
                    Since we only support capturing in the default landscape orientation, these values are just duplicates
```

### Upload Dataset step

After a session is complete, data is uploaded to the storage service using a REST PUT API.  The REST URL looks suspiciously like the file path:

```
PUT /API/DeepData/200407/%DeviceSku%/%PlainTextEmailAddress%/%SessionId%/%FileName%
```
e.g.

```
PUT https://deepeyes-wa.teamgleason.org/API/DeepData/200407/Surface_Pro_6_1796_Commercial/jbeavers%40microsoft.com/2020-07-10T02%3A22%3A12/00067-00023.jpg
PUT https://deepeyes-wa.teamgleason.org/API/DeepData/200407/Surface_Pro_6_1796_Commercial/jbeavers%40microsoft.com/2020-07-10T02%3A22%3A12/00067-00023.json
```

you can also upload per session metadata files using the same approach, e.g.

```
PUT https://deepeyes-wa.teamgleason.org/API/DeepData/200407/Surface_Pro_6_1796_Commercial/jbeavers%40microsoft.com/2020-07-10T02%3A22%3A12/deviceMetrics.json
PUT https://deepeyes-wa.teamgleason.org/API/DeepData/200407/Surface_Pro_6_1796_Commercial/jbeavers%40microsoft.com/2020-07-10T02%3A22%3A12/screen.json
```

### Prepare Dataset step

This step using facial feature recognition to identify the face and eye bounding boxes and extract the face and eyes images.  It also calculates the camera distance offsets (dotCam) using screen metrics and device + orientation to camera position lookup table.

Since we are only going to support capture and playback on 'identical' devices for now in a singular orientation, we can optionally skip the dotCam calculation step.

```
info.json       Updates NumFaceDetections and NumEyeDetections based on dlib results

faceGrid.json
dlibFace.json
dlibLeftEyeGrid.json
dlibRightEyeGrid.json

appleFace/{frameId}.jpg
appleLeftEye/{frameId}.jpg
appleRightEye/{frameId}.jpg
```

## Changelog

2021-08-18

* Noted that session metadata files as optional, as device model is now stored in the folder path
* And other metadata can be inferred from the device model, orientation is not variable in our
* capture app, and we only have one model of eye tracker in use

2020-07-27

* Restructured how JSON is rendered to clarify
* Added definition of how %frame%.json Confidence is calculated
* Added more example files
* Added upload API and example
