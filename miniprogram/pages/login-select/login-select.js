Page({
  goToPhoneLogin: function() {
    wx.navigateTo({
      url: '/pages/login/login'
    })
  },
  
  goToWechatLogin: function() {
    wx.showToast({
      title: '微信登录中...',
      icon: 'loading',
      duration: 1500
    })
    
    setTimeout(() => {
      wx.setStorageSync('userToken', 'mock_wechat_token')
      wx.setStorageSync('userInfo', { name: '微信用户', avatar: '' })
      wx.removeStorageSync('imagePath')
      wx.removeStorageSync('showPreview')
      wx.removeStorageSync('tempImage')
      wx.clearStorageSync()
      wx.setStorageSync('userToken', 'mock_wechat_token')
      wx.setStorageSync('userInfo', { name: '微信用户', avatar: '' })
      wx.redirectTo({
        url: '/pages/index/index'
      })
    }, 1500)
  }
})