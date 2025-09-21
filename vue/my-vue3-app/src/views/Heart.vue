<template>
  <div id="particles-js" class="background">
    <!-- 其他代码 -->
    <div class="heart-container">
    <div
      class="heart"
      :style="{ backgroundColor: currentColor }"
    ></div>
  </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

// 颜色数组
const colors = [
  '#ff4d6d', // 粉红
  '#ff758f', // 浅粉
  '#ff9aa2', // 淡粉
  '#ffb3c6', // 更淡粉
  '#ffccd5', // 极淡粉
  '#ff6b6b', // 珊瑚红
  '#ff8e8e', // 淡珊瑚
  '#ff5252', // 鲜红
  '#ff1744', // 亮红
  '#d50000', // 深红
]

const currentColor = ref(colors[0])
let colorIndex = 0
let intervalId = null

onMounted(() => {
  intervalId = setInterval(() => {
    colorIndex = (colorIndex + 1) % colors.length
    currentColor.value = colors[colorIndex]
  }, 800) // 每 0.8 秒换一次颜色
})

onUnmounted(() => {
  if (intervalId) clearInterval(intervalId)
})
</script>

<style scoped>
.background {
  position: absolute;
  width: 100%;
  height: 100%;
  overflow: hidden;
  top: 0;
  bottom: 0;
  left: 0;
  right: 0;
}
.heart-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #121212;
}

.heart {
  width: 100px;
  height: 90px;
  position: relative;
  animation: heartbeat 1.2s ease-in-out infinite;
}

.heart:before,
.heart:after {
  content: '';
  position: absolute;
  width: 52px;
  height: 80px;
  border-radius: 50px 50px 0 0;
  background: currentColor;
}

.heart:before {
  left: 50px;
  transform: rotate(-45deg);
  transform-origin: 0 100%;
}

.heart:after {
  left: 0;
  transform: rotate(45deg);
  transform-origin: 100% 100%;
}

@keyframes heartbeat {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.2);
  }
}
</style>