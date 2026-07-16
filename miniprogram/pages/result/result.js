Page({
  data: {
    status: '',
    hasResult: false,
    result: '',
    resultLabel: '',
    hasCancer: false,
    confidence: 0,
    probabilities: { normal: 0, benign: 0, malignant: 0 },
    imagePath: '',
    maskUrl: '',
    modelVersion: '',
    device: '',
    analysis: '',
    suggestion: '',
    message: '',
    summaryContent: ''
  },

  onLoad: function(options) {
    if (options && options.data) {
      try {
        var data = JSON.parse(decodeURIComponent(options.data))
        var hasResult = data.status !== 'model_not_configured'
        var summaryContent = hasResult
          ? (data.analysis || ('本次检测结果显示' + (data.hasCancer ? '存在恶性特征' : '未见明显异常') + '，建议结合临床症状和其他检查结果综合判断。'))
          : (data.message || 'AI模型未配置，暂不能生成诊断结果。')
        this.setData({
          status: data.status || '',
          hasResult: hasResult,
          result: data.result || '',
          resultLabel: data.resultLabel || '',
          hasCancer: !!data.hasCancer,
          confidence: data.confidence || 0,
          probabilities: data.probabilities || { normal: 0, benign: 0, malignant: 0 },
          imagePath: data.imagePath || '',
          maskUrl: data.maskUrl || '',
          modelVersion: data.modelVersion || '',
          device: data.device || '',
          analysis: data.analysis || '',
          suggestion: data.suggestion || '',
          message: data.message || '',
          summaryContent: summaryContent
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
  },

  goToHome: function() {
    wx.redirectTo({ url: '/pages/index/index' })
  }
})
