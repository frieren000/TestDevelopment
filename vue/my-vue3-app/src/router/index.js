import { createRouter, createWebHistory } from 'vue-router'

// 导入页面组件
import Home from '../views/Home.vue'
import About from '../views/About.vue'
import UserList from '../views/UserList.vue'
import UserLogin from '../views/UserLogin.vue'
import Heart from '../views/Heart.vue'
import Test from '../views/Test.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home,
  },
  {
    path: '/about',
    name: 'About',
    component: About,
  },
  {
    path: '/users',
    name: 'UserList',
    component: UserList,
  },
  {
    path: '/login',
    name: 'UsersLogin',
    component: UserLogin,
  },
  {
    path: '/Heart',
    name: 'Heart',
    component: Heart,
  },
    {
    path: '/Test',
    name: 'Test',
    component: Test,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router