<!-- <template>
  <div>
    <h1>Vue + Django 整合示例</h1>
    <p>{{ message }}</p>
    <button @click="fetchData">获取数据</button>
  </div>

  <div id="app">
    <nav>
      <router-link to="/">首页</router-link> |
      <router-link to="/about">关于我们</router-link> |
      <router-link to="/users">用户列表</router-link>
    </nav>
    <hr />
    <!-- 路由出口：匹配的组件会渲染在这里 -->
    <!-- <router-view />
  </div>

</template> -->

<!-- <script>
import axios from 'axios'

export default {
  data() {
    return {
      message: '点击按钮获取 Django 数据'
    }
  },
  methods: {
    async fetchData() {
      try {
        const response = await axios.get('/api/search_user_info/')
        this.message = response.data.message
        console.log(response.data)
      } catch (error) {
        console.error('API 调用失败:', error)
        this.message = '请求失败，请检查 Django 是否运行'
      }
    }
  }
}
</script>

<style>
nav {
  padding: 10px;
  background: #f8f8f8;
} -->
<!-- </style> -->


<!-- src/App.vue -->
<template>
  <div id="app">
    <nav>
      <router-link to="/">首页</router-link> |
      <router-link to="/about">关于我们</router-link> |
      <router-link to="/users">用户列表</router-link>
    </nav>
    <hr />
    <!-- 路由出口：匹配的组件会渲染在这里 -->
    <router-view />
  </div>
</template>

<style>
nav {
  padding: 10px;
  background: #f8f8f8;
}
</style>