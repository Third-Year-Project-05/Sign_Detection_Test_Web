import React, { useCallback, useEffect, useRef, useState } from "react";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";

const HolisticComponent = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const keypointsSequenceRef = useRef([]);
  const [isBackendEnabled, setIsBackendEnabled] = useState(true); // Toggle state

  // Backend URL
  const backendURL = "http://127.0.0.1:9000/predict";

  const sendToBackend = useCallback(async (keypointsSequence) => {
    try {
      const response = await fetch(backendURL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ keypoint: keypointsSequence }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Prediction:", data);
      } else {
        console.error("Error in backend response", response.statusText);
      }
    } catch (error) {
      console.error("Error connecting to backend", error);
    }
  }, []);

  // Create a landmarks array
  const flattenLandmarks = useCallback((landmarks, expectedLength) => {
    if (landmarks) {
      const flattened = landmarks.flatMap((point) => [
        point.x,
        point.y,
        point.z,
      ]);

      if (flattened.length > expectedLength) {
        return flattened.slice(0, expectedLength);
      }
      return [
        ...flattened,
        ...Array(expectedLength - flattened.length).fill(0),
      ];
    }
    return Array(expectedLength).fill(0);
  }, []);

  // Extract keypoints
  const extractKeypoints = useCallback(
    (results) => {
      const pose = results.poseLandmarks
        ? results.poseLandmarks.flatMap((point) => [
            point.x,
            point.y,
            point.z,
            point.visibility,
          ])
        : Array(33 * 4).fill(0);

      const face = flattenLandmarks(results.faceLandmarks, 468 * 3);
      const leftHand = flattenLandmarks(results.leftHandLandmarks, 21 * 3);
      const rightHand = flattenLandmarks(results.rightHandLandmarks, 21 * 3);

      return [...pose, ...face, ...leftHand, ...rightHand];
    },
    [flattenLandmarks]
  );

  // Draw the camera feed on the canvas
  const drawCameraFeed = () => {
    const canvasCtx = canvasRef.current.getContext("2d");
    canvasCtx.clearRect(
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height
    );

    canvasCtx.drawImage(
      videoRef.current,
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height
    );
  };

  useEffect(() => {
    const holistic = new Holistic({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    holistic.onResults((results) => {
      if (isBackendEnabled) {
        if (
          (!results.leftHandLandmarks ||
            results.leftHandLandmarks.length === 0) &&
          (!results.rightHandLandmarks ||
            results.rightHandLandmarks.length === 0)
        ) {
          console.log("No hand landmarks detected, skipping...");
          // camer preview com
          drawCameraFeed();
          return;
        }

        const keypoints = extractKeypoints(results);
        keypointsSequenceRef.current.push(keypoints);

        if (keypointsSequenceRef.current.length > 30) {
          keypointsSequenceRef.current =
            keypointsSequenceRef.current.slice(-30);
        }

        if (keypointsSequenceRef.current.length === 30) {
          sendToBackend(keypointsSequenceRef.current);
        }
      }

      drawCameraFeed();
    });

    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        if (isBackendEnabled) {
          await holistic.send({ image: videoRef.current });
        } else {
          drawCameraFeed(); // Draw camera feed without processing
        }
      },
      width: 1280,
      height: 720,
    });

    camera.start();

    return () => {
      camera.stop();
    };
  }, [isBackendEnabled, sendToBackend, extractKeypoints]);

  return (
    <div style={{ position: "relative" }}>
      <video ref={videoRef} style={{ display: "none" }} />
      <canvas
        ref={canvasRef}
        width="1280"
        height="720"
        style={{
          width: "100%",
          height: "auto",
          border: "1px solid black",
        }}
      ></canvas>
      <div style={{ marginTop: "10px", textAlign: "center" }}>
        <button onClick={() => setIsBackendEnabled((prev) => !prev)}>
          {isBackendEnabled ? "Disable Backend" : "Enable Backend"}
        </button>
      </div>
    </div>
  );
};

export default HolisticComponent;
