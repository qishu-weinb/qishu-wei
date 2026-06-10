Page({
  data: {
    phone: '',
    password: '',
    showPassword: false,
    loading: false
  },

  goBack: function() {
    wx.navigateBack()
  },

  onPhoneInput: function(e) {
    this.setData({ phone: e.detail.value })
  },

  onPasswordInput: function(e) {
    this.setData({ password: e.detail.value })
  },

  togglePassword: function() {
    this.setData({ showPassword: !this.data.showPassword })
  },

  handleLogin: function() {
    if (!this.data.phone) {
      wx.showToast({ title: '请输入手机号', icon: 'none' })
      return
    }
    if (!this.data.password) {
      wx.showToast({ title: '请输入密码', icon: 'none' })
      return
    }
    
    this.setData({ loading: true })
    
    setTimeout(() => {
      wx.setStorageSync('userToken', 'mock_token')
      wx.setStorageSync('userInfo', { name: '用户', avatar: '' })
      wx.redirectTo({
        url: '/pages/index/index'
      })
    }, 500)
  },

  goToForget: function() {
    wx.showToast({ title: '忘记密码功能开发中', icon: 'none' })
  },

  goToRegister: function() {
    wx.navigateTo({
      url: '/pages/register/register'
    })
  }
})