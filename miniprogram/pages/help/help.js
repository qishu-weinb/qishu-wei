Page({
  data: {
    expandedIndex: -1
  },

  onLoad: function(options) {
    if (!wx.getStorageSync('userToken')) {
      wx.redirectTo({ url: '/pages/login-select/login-select' })
    }
  },

  goBack: function() {
    wx.navigateBack()
  },

  toggleFaq: function(index) {
    this.setData({
      expandedIndex: this.data.expandedIndex === index ? -1 : index
    })
  }
})
