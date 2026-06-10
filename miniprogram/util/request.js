const BASE_URL = 'http://localhost:5000/api'

const request = (options) => {
  const {
    url,
    method = 'GET',
    data = {},
    header = {},
    loading = true,
    loadingText = '加载中...'
  } = options

  return new Promise((resolve, reject) => {
    if (loading) {
      wx.showLoading({
        title: loadingText,
        mask: true
      })
    }

    const token = wx.getStorageSync('userToken')
    const defaultHeader = {
      'Content-Type': 'application/json',
      'Authorization': token ? `Bearer ${token}` : ''
    }

    wx.request({
      url: BASE_URL + url,
      method: method.toUpperCase(),
      data: data,
      header: { ...defaultHeader, ...header },
      success: (res) => {
        if (loading) {
          wx.hideLoading()
        }

        const { statusCode, data: responseData } = res

        if (statusCode === 200) {
          const { code, message, data } = responseData

          if (code === 0) {
            resolve(data)
          } else if (code === 401) {
            wx.removeStorageSync('userToken')
            wx.removeStorageSync('userInfo')
            wx.redirectTo({ url: '/pages/login/login' })
            reject(new Error(message || '登录已过期'))
          } else {
            wx.showToast({ title: message || '请求失败', icon: 'none' })
            reject(new Error(message || '请求失败'))
          }
        } else {
          wx.showToast({ title: '网络请求失败', icon: 'none' })
          reject(new Error(`HTTP错误: ${statusCode}`))
        }
      },
      fail: (err) => {
        if (loading) {
          wx.hideLoading()
        }
        wx.showToast({ title: '网络异常', icon: 'none' })
        reject(err)
      }
    })
  })
}

const api = {
  login: (data) => request({ url: '/login', method: 'POST', data }),
  register: (data) => request({ url: '/register', method: 'POST', data }),
  logout: () => request({ url: '/logout', method: 'POST' }),
  getUserInfo: () => request({ url: '/user/info', method: 'GET' }),
  updateUserInfo: (data) => request({ url: '/user/info', method: 'PUT', data }),
  uploadImage: (data) => request({ 
    url: '/upload/image', 
    method: 'POST', 
    data,
    header: { 'Content-Type': 'multipart/form-data' }
  }),
  getDiagnosisResult: (imageId) => request({ url: `/diagnosis/${imageId}`, method: 'GET' }),
  getHistoryList: (params) => request({ url: '/history', method: 'GET', data: params }),
  deleteHistory: (id) => request({ url: `/history/${id}`, method: 'DELETE' }),
  getKnowledgeList: (params) => request({ url: '/knowledge', method: 'GET', data: params }),
  getKnowledgeDetail: (id) => request({ url: `/knowledge/${id}`, method: 'GET' })
}

module.exports = {
  request,
  api
}