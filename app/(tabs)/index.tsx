import React, { useState, useEffect, useRef } from "react";
import { StyleSheet, View, Text, Button, TouchableOpacity } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as FileSystem from "expo-file-system";

import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { CameraType } from "expo-camera/build/legacy/Camera.types";

interface Prediction {
  class: string;
  score: number;
  bbox: [number, number, number, number];
}

interface CarbonEmissions {
  [key: string]: number;
}

export default function HomeScreen() {
  const [cameraType, setCameraType] = useState<CameraType>(CameraType.back);
  const [permission, requestPermission] = useCameraPermissions();
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [carbonEmissions, setCarbonEmissions] = useState<CarbonEmissions>({});
  const cameraRef = useRef<CameraView | null>(null);

  useEffect(() => {
    setTimeout(() => {
      loadModel();
    }, 3000);
  }, []);

  const loadModel = async () => {
    await tf.ready();
    const model = await cocoSsd.load();
    detectObjects(model);
  };

  const detectObjects = async (model: cocoSsd.ObjectDetection) => {
    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          base64: true,
        });
        if (photo?.base64) {
          const imageData = await convertBase64ToUint8Array(photo.base64);
          const imageTensor = tf.tensor3d(imageData, [480, 640, 3]);
  
          const normalizedTensor = imageTensor.toFloat().div(tf.scalar(255));
          
          const reshapedTensor = normalizedTensor.reshape([1, 480, 640, 3]) as tf.Tensor3D;  
          const predictions = await model.detect(reshapedTensor);
          setPredictions(predictions);
          updateCarbonEmissions(predictions);
          
          imageTensor.dispose();
          normalizedTensor.dispose();
          reshapedTensor.dispose();
        }
      } catch (error) {
        console.error("Error in object detection:", error);
      }
    }
    requestAnimationFrame(() => detectObjects(model));
  };
  
  
  const convertBase64ToUint8Array = async (base64: string): Promise<Uint8Array> => {
    try {
      if (!base64 || typeof base64 !== 'string') {
        throw new Error('Invalid base64 string');
      }
      
      const cleanedBase64 = base64.replace(/^data:image\/\w+;base64,/, '');
      const trimmedBase64 = cleanedBase64.replace(/\s+/g, '');
      
      if (!/^[A-Za-z0-9+/=]+$/.test(trimmedBase64)) {
        throw new Error('Invalid base64 string');
      }
      
      const binaryString = atob(trimmedBase64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
  
      const reshapedBytes = new Uint8Array(480 * 640 * 3);
      for (let y = 0; y < 480; y++) {
        for (let x = 0; x < 640; x++) {
          for (let c = 0; c < 3; c++) {
            reshapedBytes[(y * 640 + x) * 3 + c] = bytes[(y * 640 + x) * 4 + c];
          }
        }
      }
  
      return reshapedBytes;
    } catch (error) {
      console.error('Error decoding base64 string:', error);
      throw error;
    }
  };


  const updateCarbonEmissions = (predictions: Prediction[]) => {
    const emissions = predictions.reduce(
      (acc, curr) => ({
        ...acc,
        [curr.class]: calculateEmissions(curr.class),
      }),
      {} as CarbonEmissions
    );
    setCarbonEmissions(emissions);
  };

  const calculateEmissions = (className: string): number => {
    return Math.random() * 100;
  };

  function toggleCameraType() {
    setCameraType((current) =>
      current === CameraType.back ? CameraType.front : CameraType.back
    );
    console.log("executed togglecamera");
  }

  if (!permission) {
    return (
      <ThemedView>
        <ThemedText>Requesting camera permission...</ThemedText>
      </ThemedView>
    );
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet
    return (
      <ThemedView style={styles.container}>
        <ThemedText style={styles.message}>
          We need your permission to show the camera
        </ThemedText>
        <Button onPress={requestPermission} title="Grant permission" />
      </ThemedView>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera}>
        {/* Camera controls */}
        {predictions.map((prediction, index) => (
          <View
            key={index}
            style={[
              styles.prediction,
              {
                top: prediction.bbox[1],
                left: prediction.bbox[0],
                width: prediction.bbox[2] - prediction.bbox[0],
                height: prediction.bbox[3] - prediction.bbox[1],
              },
            ]}
          >
            <ThemedText type="subtitle">{prediction.class}</ThemedText>
            <ThemedText type="defaultSemiBold">
              Carbon Emissions: {carbonEmissions[prediction.class]?.toFixed(2)}%
            </ThemedText>
          </View>
        ))}
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
  },
  message: {
    textAlign: "center",
    paddingBottom: 10,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: "row",
    backgroundColor: "transparent",
    margin: 64,
  },
  button: {
    flex: 1,
    alignSelf: "flex-end",
    alignItems: "center",
  },
  text: {
    fontSize: 24,
    fontWeight: "bold",
    color: "white",
  },
  prediction: {
    position: "absolute",
    backgroundColor: "rgba(255, 255, 255, 0.5)",
    padding: 8,
    borderWidth: 2,
    borderColor: "red",
  },
});
