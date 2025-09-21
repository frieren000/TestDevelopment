<template>
    <div>
        <h2>系统登录</h2>
        <p>请输入您的账号密码登录系统</p>
    </div>
    <el-form
        ref="loginFormRef"
            :model="form"
            :rules="rules"
            label-position="top"
            @submit.prevent="handleLogin"
            >
         <el-form-item prop="username" label="用户名">
          <el-input
            v-model="form.username"
            placeholder="请输入用户名"
            prefix-icon="User"
            clearable
            @blur="validateField('username')"
          />
        </el-form-item>
    </el-form>
</template>
<script setup></script>
<style></style>