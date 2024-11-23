import React, { useCallback, useEffect, useRef } from "react";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";

const HolisticComponent = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const keypointsSequenceRef = useRef([]);

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
        console.log("Prediction:", data.prediction);
      } else {
        console.error("Error in backend response", response.statusText);
      }
    } catch (error) {
      console.error("Error connecting to backend", error);
    }
  }, []);

  // create a landmarks array
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

  // return a image for landmark result array
  const extractKeypoints = useCallback(
    (results) => {
      const pose = results.poseLandmarks
        ? results.poseLandmarks.flatMap((point) => [
            point.x,
            point.y,
            point.z,
            point.visibility,
          ])
        : Array(33 * 3).fill(0);

      // Face: 468 landmarks, each with (x, y, z)
      const face = flattenLandmarks(results.faceLandmarks, 468 * 3);
      // Left and Right hand: 21 landmarks, each with (x, y, z)
      const leftHand = flattenLandmarks(results.leftHandLandmarks, 21 * 3);
      const rightHand = flattenLandmarks(results.rightHandLandmarks, 21 * 3);

      // Concatenate all parts into a single array
      return [...pose, ...face, ...leftHand, ...rightHand];
    },
    [flattenLandmarks]
  );

  // Function to create a canvas
  const createCanvas = () => {
    const canvasCtx = canvasRef.current.getContext("2d");
    canvasCtx.clearRect(
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height
    );

    canvasCtx.save();
    canvasCtx.drawImage(
      videoRef.current,
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height
    );

    // Optionally, draw landmarks or other visual elements here.
    canvasCtx.restore();
  };

  useEffect(() => {
    // Initialize Mediapipe Holistic
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

    // Callback for Mediapipe results
    holistic.onResults((results) => {
      const keypoints = extractKeypoints(results);
      // console.log(keypoints);
      keypointsSequenceRef.current.push(keypoints);

      // Keep only the last 30 frames
      if (keypointsSequenceRef.current.length > 30) {
        keypointsSequenceRef.current = keypointsSequenceRef.current.slice(-30);
      }

      if (keypointsSequenceRef.current.length === 30) {
        setInterval(() => {
          sendToBackend(keypointsSequenceRef.current);
        }, 16.7);
      }
      createCanvas();
    });

    // Setup camera
    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        await holistic.send({ image: videoRef.current });
      },
      width: 1280,
      height: 720,
    });

    camera.start();

    // Cleanup on unmount
    return () => {
      camera.stop();
    };
  }, [sendToBackend, extractKeypoints]);

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
    </div>
  );
};

export default HolisticComponent;
