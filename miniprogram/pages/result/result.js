Page({
  data: {
    hasCancer: false,
    confidence: 0,
    advice: ''
  },

  onLoad: function(options) {
    if (options && options.data) {
      try {
        var data = JSON.parse(decodeURIComponent(options.data))
        this.setData({
          hasCancer: data.hasCancer,
          confidence: data.confidence
        })
      } catch (e) {
        console.error('解析数据失败', e)
      }
    }
  },

  goBack: function() {
    wx.navigateBack()
  },

  goToExpert: function() {
    wx.showToast({ title: '专家解读功能开发中', icon: 'none' })
  }
})