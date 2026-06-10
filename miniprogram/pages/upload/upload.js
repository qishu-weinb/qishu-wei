const { uploadDiagnosisImage } = require('../../util/request')

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
    wx.showLoading({ title: '检测中...' })

    uploadDiagnosisImage(this.data.imagePath).then(function(response) {
      wx.hideLoading()
      var data = response.data || {}
      var resultData = {
        status: data.status || 'completed',
        recordId: data.recordId || data.id || '',
        imageId: data.imageId || '',
        imageUrl: data.imageUrl || '',
        hasCancer: data.result === 'malignant',
        confidence: data.confidence || 0,
        analysis: data.analysis || '',
        suggestion: data.suggestion || '',
        message: response.message || ''
      }
      wx.redirectTo({
        url: '/pages/result/result?data=' + encodeURIComponent(JSON.stringify(resultData))
      })
    }).catch(function() {
      wx.hideLoading()
    })
  }
})
