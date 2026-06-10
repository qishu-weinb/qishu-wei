const { api } = require('../../util/request')

Page({
  data: {
    phone: '',
    password: '',
    confirmPassword: '',
    showPassword: false,
    showConfirmPassword: false,
    loading: false
  },

  onUsernameInput: function(e) {
    this.setData({ phone: e.detail.value })
  },

  onPasswordInput: function(e) {
    this.setData({ password: e.detail.value })
  },

  onConfirmPasswordInput: function(e) {
    this.setData({ confirmPassword: e.detail.value })
  },

  togglePassword: function() {
    this.setData({ showPassword: !this.data.showPassword })
  },

  toggleConfirmPassword: function() {
    this.setData({ showConfirmPassword: !this.data.showConfirmPassword })
  },

  handleRegister: function() {
    var phone = this.data.phone
    var password = this.data.password
    var confirmPassword = this.data.confirmPassword
    
    if (!phone || !password || !confirmPassword) {
      wx.showToast({ title: '请填写完整信息', icon: 'none' })
      return
    }
    
    if (password !== confirmPassword) {
      wx.showToast({ title: '两次输入的密码不一致', icon: 'none' })
      return
    }

    this.setData({ loading: true })

    api.register({
      phone: phone,
      password: password,
      confirmPassword: confirmPassword
    }).then(() => {
      wx.showToast({ title: '注册成功', icon: 'success' })
      setTimeout(function() {
        wx.navigateBack()
      }, 1500)
    }).finally(() => {
      this.setData({ loading: false })
    })
  },

  goToLogin: function() {
    wx.navigateBack()
  }
})
