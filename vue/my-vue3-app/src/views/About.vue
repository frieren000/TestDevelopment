<template>
  <div class="about-page">
    <h1>关于我们</h1>
    <p>这是“关于我们”页面的内容。</p>
    <router-link to="/">返回首页</router-link>
  </div>
</template>

<script>
export default {
  name: 'AboutPage'
}
</script>

<style scoped>
.about-page {
  padding: 20px;
  text-align: center;
}
</style>