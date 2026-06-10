Page({
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