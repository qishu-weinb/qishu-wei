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

  toggleFaq: function(e) {
    var index = Number(e.currentTarget.dataset.index)
    this.setData({
      expandedIndex: this.data.expandedIndex === index ? -1 : index
    })
  }
})
