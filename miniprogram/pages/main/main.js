Page({
  data: {
    loading: false
  },

  onLoad: function(options) {
    if (!wx.getStorageSync('userToken')) {
      wx.redirectTo({ url: '/pages/index/index' })
    }
  },

  selectImage: function() {
    wx.navigateTo({
      url: '/pages/upload/upload'
    })
  },

  goToHistory: function() {
    wx.showToast({ title: '历史记录功能开发中', icon: 'none' })
  },

  goToGuide: function() {
    wx.navigateTo({
      url: '/pages/guide/guide'
    })
  },

  goToKnowledge: function() {
    wx.navigateTo({
      url: '/pages/knowledge/knowledge'
    })
  },

  goToReport: function() {
    wx.showToast({ title: '检测报告功能开发中', icon: 'none' })
  },

  goToAbout: function() {
    wx.showToast({ title: '关于我们功能开发中', icon: 'none' })
  },

  goToHelp: function() {
    wx.navigateTo({
      url: '/pages/help/help'
    })
  },

  goToScan: function() {
    this.selectImage()
  },

  goToProfile: function() {
    wx.navigateTo({
      url: '/pages/profile/profile'
    })
  }
})
