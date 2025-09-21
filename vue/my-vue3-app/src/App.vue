<template>
  <div id="app">
    <nav>
      <router-link to="/">首页</router-link> |
      <router-link to="/about">关于我们</router-link> |
      <router-link to="/users">用户列表</router-link> |
      <router-link to="/login">用户登录</router-link> |
      <router-link to="/heart">图案</router-link> |
      <router-link to="/test">测试页面</router-link>  |

    </nav>
    <hr />
    <router-view />
  </div>
</template>

<style>
nav {
  padding: 10px;
  background: #f8f8f8;
}
</style>