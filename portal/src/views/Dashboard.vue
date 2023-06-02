<template>
  <div class="dashboard">
    <img class="map-image" src="@/assets/map.webp" alt="Map">
  </div>
</template>

<style scoped>
.dashboard {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
}

.map-image {
  width: auto;
  filter: brightness(80%); /* 降低亮度 */
  opacity: 0.8; /* 控制透明度 */
}
</style>
