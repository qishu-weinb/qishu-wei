const { api } = require('../../util/request')

Page({
  data: {
    articles: []
  },

  onLoad: function() {
    this.loadArticles()
  },

  loadArticles: function() {
    api.getKnowledgeList({ page: 1, size: 20 }).then((data) => {
      this.setData({ articles: data.list || [] })
    })
  },

  goBack: function() {
    wx.navigateBack()
  },
  
  goToDetail: function(e) {
    var id = e.currentTarget.dataset.id || '1'
    wx.navigateTo({
      url: '/pages/knowledge-detail/knowledge-detail?id=' + id
    })
  }
})
