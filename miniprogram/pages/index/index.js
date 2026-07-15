const { api } = require('../../util/request')

Page({
  data: {
    loading: false,
    historyList: []
  },

  goBack: function() {
    wx.navigateBack()
  },

  clearImage: function() {
    wx.showToast({ title: '已清空选择', icon: 'none' })
  },

  onLoad: function(options) {
    if (!wx.getStorageSync('userToken')) {
      wx.redirectTo({ url: '/pages/login-select/login-select' })
      return
    }
    this.loadRecentHistory()
  },

  loadRecentHistory: function() {
    api.getHistoryList({ page: 1, size: 3 }).then((data) => {
      var list = (data.list || []).map(function(item) {
        var pending = item.status === 'model_not_configured'
        return {
          ...item,
          icon: pending ? '⏳' : (item.result === 'malignant' ? '⚠️' : '✅'),
          iconClass: item.result === 'malignant' ? 'malignant' : 'benign'
        }
      })
      this.setData({ historyList: list })
    })
  },

  selectImage: function() {
    wx.navigateTo({
      url: '/pages/upload/upload'
    })
  },

  goToHistory: function() {
    this.loadRecentHistory()
    wx.showToast({ title: '已刷新历史记录', icon: 'none' })
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
  },

  handleLogout: function() {
    api.logout().finally(() => {
      wx.removeStorageSync('userToken')
      wx.removeStorageSync('userInfo')
      wx.redirectTo({
        url: '/pages/login/login'
      })
    })
  }
})
