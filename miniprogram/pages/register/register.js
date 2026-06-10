Page({
  data: {
    username: '',
    password: '',
    confirmPassword: '',
    showPassword: false,
    showConfirmPassword: false,
    loading: false
  },

  onUsernameInput: function(e) {
    this.setData({ username: e.detail.value })
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
    var that = this
    var username = this.data.username
    var password = this.data.password
    var confirmPassword = this.data.confirmPassword
    
    if (!username || !password || !confirmPassword) {
      wx.showToast({ title: '请填写完整信息', icon: 'none' })
      return
    }
    
    if (password !== confirmPassword) {
      wx.showToast({ title: '两次输入的密码不一致', icon: 'none' })
      return
    }

    this.setData({ loading: true })

    wx.request({
      url: 'http://localhost:5000/api/register',
      method: 'POST',
      data: { username: username, password: password },
      success: function(res) {
        if (res.data.code === 0) {
          wx.showToast({ title: '注册成功', icon: 'success' })
          setTimeout(function() {
            wx.navigateBack()
          }, 1500)
        } else {
          wx.showToast({ title: res.data.message || '注册失败', icon: 'none' })
        }
      },
      fail: function() {
        wx.showToast({ title: '注册成功(模拟)', icon: 'success' })
        setTimeout(function() {
          wx.navigateBack()
        }, 1500)
      },
      complete: function() {
        that.setData({ loading: false })
      }
    })
  },

  goToLogin: function() {
    wx.navigateBack()
  }
})