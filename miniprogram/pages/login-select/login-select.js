const { api } = require('../../util/request')

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

    wx.login({
      success: (res) => {
        api.loginWechat({ code: res.code }).then((data) => {
          wx.setStorageSync('userToken', data.token)
          wx.setStorageSync('userInfo', data.userInfo)
          wx.redirectTo({
            url: '/pages/index/index'
          })
        })
      },
      fail: () => {
        wx.showToast({ title: '微信登录失败', icon: 'none' })
      }
    })
  }
})
