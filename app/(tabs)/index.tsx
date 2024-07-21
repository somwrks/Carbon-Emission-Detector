import { Image, Text, View, TouchableOpacity } from "react-native";
import { HelloWave } from "@/components/HelloWave";
import ParallaxScrollView from "@/components/ParallaxScrollView";
import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";

export default function HomeScreen() {
  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: "#A1CEDC", dark: "#1D3D47" }}
      headerImage={
        <Image
          source={require("@/assets/images/partial-react-logo.png")}
          style={{
            height: 178,
            width: 290,
            position: "absolute",
            bottom: 0,
            left: 0,
          }}
        />
      }
    >
      <ThemedView className="flex-row items-center gap-2 p-4">
        <ThemedText className="text-4xl font-bold">
          Carbon Emission Detector
        </ThemedText>
      </ThemedView>
      
      <View className="p-4">
        <TouchableOpacity className="bg-blue-500 p-2 rounded">
          <Text className="text-white">Hello</Text>
        </TouchableOpacity>
      </View>
    </ParallaxScrollView>
  );
}
