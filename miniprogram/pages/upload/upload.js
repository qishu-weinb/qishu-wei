Page({
  data: {
    imagePath: ''
  },

  onLoad: function(options) {
    var imagePath = wx.getStorageSync('uploadImage')
    this.setData({ imagePath: imagePath || '' })
    wx.removeStorageSync('uploadImage')
  },

  goBack: function() {
    wx.navigateBack()
  },

  selectImage: function() {
    var that = this
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'],
      sourceType: ['album', 'camera'],
      success: function(res) {
        that.setData({ imagePath: res.tempFilePaths[0] })
      }
    })
  },

  clearImage: function() {
    this.setData({ imagePath: '' })
  },

  handleUpload: function() {
    if (!this.data.imagePath) {
      wx.showToast({ title: '请先选择图片', icon: 'none' })
      return
    }
    
    var that = this
    wx.showLoading({ title: '检测中...' })
    
    setTimeout(function() {
      wx.hideLoading()
      var mockData = {
        hasCancer: Math.random() > 0.5,
        confidence: Math.floor(Math.random() * 30) + 70,
        detectTime: new Date().toLocaleString('zh-CN')
      }
      wx.redirectTo({
        url: '/pages/result/result?data=' + encodeURIComponent(JSON.stringify(mockData))
      })
    }, 1500)
  }
})
