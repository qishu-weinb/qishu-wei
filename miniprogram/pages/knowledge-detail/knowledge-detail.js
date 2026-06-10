const { api } = require('../../util/request')

Page({
  data: {
    article: {
      tag: '',
      date: '',
      title: '',
      content: ''
    }
  },

  onLoad: function(options) {
    var articleId = options.id || '1'
    this.loadArticle(articleId)
  },

  loadArticle: function(id) {
    api.getKnowledgeDetail(id).then((article) => {
      this.setData({
        article: article
      })
    })
  },

  goBack: function() {
    wx.navigateBack()
  }
})
