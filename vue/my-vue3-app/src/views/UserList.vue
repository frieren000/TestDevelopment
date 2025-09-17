<!-- src/views/UserList.vue -->
<template>
  <div class="user-list">
    <h1>用户列表</h1>
    <ul>
      <li v-for="user in users" :key="user.id">
        {{ user.name }} - {{ user.email }}
      </li>
    </ul>
    <router-link to="/">返回首页</router-link>
  </div>
</template>

<script>
export default {
  name: 'UserList',
  data() {
    return {
      users: [
        { id: 1, name: '张三', email: 'zhang@example.com' },
        { id: 2, name: '李四', email: 'li@example.com' }
      ]
    }
  }
}
</script>

<style scoped>
.user-list {
  padding: 20px;
}
</style>