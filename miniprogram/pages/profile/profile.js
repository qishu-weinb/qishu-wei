Page({
  data: {
    userInfo: {
      name: '用户昵称',
      id: '12345678',
      avatar: ''
    },
    stats: {
      totalTests: 12,
      benignResults: 8,
      suspiciousResults: 4
    }
  },

  onLoad: function(options) {
    if (!wx.getStorageSync('userToken')) {
      wx.redirectTo({ url: '/pages/login-select/login-select' })
    }
  },

  goBack: function() {
    wx.navigateBack()
  },

  goToHistory: function() {
    wx.showToast({ title: '检测历史功能开发中', icon: 'none' })
  },

  goToReport: function() {
    wx.showToast({ title: '检测报告功能开发中', icon: 'none' })
  },

  goToFavorite: function() {
    wx.showToast({ title: '收藏夹功能开发中', icon: 'none' })
  },

  goToSettings: function() {
    wx.showToast({ title: '系统设置功能开发中', icon: 'none' })
  },

  goToFeedback: function() {
    wx.showToast({ title: '意见反馈功能开发中', icon: 'none' })
  },

  goToAbout: function() {
    wx.showToast({ title: '关于我们功能开发中', icon: 'none' })
  },

  logout: function() {
    wx.showModal({
      title: '确认退出',
      content: '确定要退出登录吗？',
      success: function(res) {
        if (res.confirm) {
          wx.removeStorageSync('userToken')
          wx.redirectTo({ url: '/pages/login-select/login-select' })
        }
      }
    })
  }
})
