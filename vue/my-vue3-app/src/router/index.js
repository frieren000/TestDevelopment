import { createRouter, createWebHistory } from 'vue-router'

// 导入页面组件
import Home from '../views/Home.vue'
import About from '../views/About.vue'
import UserList from '../views/UserList.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/about',
    name: 'About',
    component: About
  },
  {
    path: '/users',
    name: 'UserList',
    component: UserList
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router