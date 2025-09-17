<template>
  <div class="login-container">
    <el-card class="login-card" shadow="never">
      <template #header>
        <div class="text-center">
          <h2 class="text-2xl font-bold text-gray-800 mb-1">系统登录</h2>
          <p class="text-gray-500 text-sm">请输入您的账号密码登录系统</p>
        </div>
      </template>

      <el-form
        ref="loginFormRef"
        :model="form"
        :rules="rules"
        label-position="top"
        @submit.prevent="handleLogin"
        class="login-form"
      >
        <!-- 用户名 -->
        <el-form-item prop="username" label="用户名">
          <el-input
            v-model="form.username"
            placeholder="请输入用户名"
            prefix-icon="User"
            clearable
            @blur="validateField('username')"
          />
        </el-form-item>

        <!-- 密码 -->
        <el-form-item prop="password" label="密码">
          <el-input
            v-model="form.password"
            type="password"
            placeholder="请输入密码"
            prefix-icon="Lock"
            show-password
            @blur="validateField('password')"
          />
        </el-form-item>

        <!-- 记住我 + 忘记密码 -->
        <div class="remember-forgot-wrapper mb-6">
          <div class="flex items-center gap-2 flex-wrap justify-between">
            <el-checkbox v-model="form.remember" size="small">记住我</el-checkbox>
            <el-link type="primary" :underline="false" @click="goToForgotPassword">忘记密码？</el-link>
          </div>
        </div>

        <!-- 登录按钮 -->
        <el-form-item>
          <el-button
            type="primary"
            size="large"
            class="w-full"
            :loading="loading"
            @click="handleLogin"
          >
            <span v-if="!loading">登 录</span>
            <span v-else>登录中...</span>
          </el-button>
        </el-form-item>
      </el-form>

      <!-- 注册提示 -->
      <div class="register-prompt text-center mt-4">
        <span class="text-gray-500">还没有账号？</span>
        <span class="mx-1"></span>
        <el-link type="primary" :underline="false" @click="goToRegister">立即注册</el-link>
      </div>

      <!-- 全局错误提示 -->
      <el-alert
        v-if="globalError"
        :title="globalError"
        type="error"
        show-icon
        class="mt-4"
        @close="globalError = ''"
        close-text="关闭"
      />
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

const loginFormRef = ref(null)

const form = reactive({
  username: '',
  password: '',
  remember: false
})

const rules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 20, message: '用户名长度在 3 到 20 个字符', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码不能少于6位', trigger: 'blur' }
  ]
}

const loading = ref(false)
const globalError = ref('')
const router = useRouter()

const validateField = (prop) => {
  if (loginFormRef.value) {
    loginFormRef.value.validateField(prop)
  }
}

const handleLogin = async () => {
  if (!loginFormRef.value) return

  loginFormRef.value.validate(async (valid) => {
    if (!valid) {
      ElMessage.warning('请检查输入内容')
      return 
    }

    loading.value = true
    globalError.value = ''

    try {
      await new Promise(resolve => setTimeout(resolve, 1500))

      if (form.username !== 'admin' || form.password !== '123456') {
        throw new Error('用户名或密码错误')
      }

      ElMessage.success('登录成功！')

      if (form.remember) {
        localStorage.setItem('rememberUser', JSON.stringify({
          username: form.username
        }))
      }

      router.push('/dashboard')

    } catch (error) {
      globalError.value = error.message || '登录失败，请稍后再试'
      ElMessage.error(globalError.value)

      setTimeout(() => {
        globalError.value = ''
      }, 3000)
    } finally {
      loading.value = false
    }
  })
}

const goToRegister = () => {
  router.push('/register')
}

onMounted(() => {
  const saved = localStorage.getItem('rememberUser')
  if (saved) {
    const user = JSON.parse(saved)
    form.username = user.username
    form.remember = true

    nextTick(() => {
      const passwordInput = document.querySelector('input[type="password"]')
      if (passwordInput) passwordInput.focus()
    })
  }
})

const goToForgotPassword = () => {
  router.push('/forgot-password')
}
</script>

<style scoped>
.login-container {
    height: auto;
    max-height: 100%;
    width: auto;
    max-width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    /* min-height: 100px; */
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 5vh 5vw;
}

.login-card {
  width: 100%;
  max-width: 100%; /* ✅ 固定最大宽度 */
  border-radius: 12px;
  background-color: #fff;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  padding: 32px; /* ✅ 增加内部间距 */
  max-height: calc(100vh - 100px); /* 设置最大高度 */
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.login-form {
  margin-top: 0.5px;
}

:deep(.el-card__header) {
  border-bottom: none;
  padding: 24px 24px 16px;
}

:deep(.el-card__body) {
  padding: 24px;
}

:deep(.el-input__wrapper) {
  border-radius: 8px;
  min-height: 44px;
}

:deep(.el-checkbox) {
  font-size: 14px;
}

:deep(.el-button.size-large) {
  min-height: 48px;
}

/* ✅ 响应式标题字体 */
.text-2xl {
  font-size: 1.5rem;
}

@media (max-width: 480px) {
  .text-2xl {
    font-size: 1.25rem;
  }
  .text-sm {
    font-size: 0.8125rem;
  }
  .login-container {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
  }
}

/* ✅ 记住我 + 忘记密码 响应式换行 */
.remember-forgot-wrapper .flex {
  gap: 8px;
}

@media (max-width: 375px) {
  .remember-forgot-wrapper .flex {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px !important;
  }
}

/* ✅ 注册提示响应式间距 */
.register-prompt .mx-1 {
  display: inline-block;
  margin: 0 4px;
}

@media (max-width: 375px) {
  .register-prompt .mx-1 {
    margin: 0 8px;
  }
}

/* ✅ 全局错误提示在小屏更醒目 */
:deep(.el-alert) {
  padding: 12px;
}

@media (max-width: 375px) {
  :deep(.el-alert) {
    padding: 16px;
  }
}
</style>