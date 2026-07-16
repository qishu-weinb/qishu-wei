// 微信开发者工具本地调试可使用 localhost；真机/上线时改成已备案的 HTTPS 域名。
const BASE_URL = 'http://localhost:5000/api'
const ASSET_BASE_URL = BASE_URL.replace(/\/api$/, '')

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
  loginWechat: (data) => request({ url: '/login/wechat', method: 'POST', data }),
  logout: () => request({ url: '/logout', method: 'POST' }),
  getUserInfo: () => request({ url: '/user/info', method: 'GET' }),
  updateUserInfo: (data) => request({ url: '/user/info', method: 'PUT', data }),
  getUserStats: () => request({ url: '/user/stats', method: 'GET' }),
  getDiagnosisResult: (imageId) => request({ url: `/diagnosis/${imageId}`, method: 'GET' }),
  getHistoryList: (params) => request({ url: '/history', method: 'GET', data: params }),
  deleteHistory: (id) => request({ url: `/history/${id}`, method: 'DELETE' }),
  getKnowledgeList: (params) => request({ url: '/knowledge', method: 'GET', data: params }),
  getKnowledgeDetail: (id) => request({ url: `/knowledge/${id}`, method: 'GET' })
}

const uploadDiagnosisImage = (filePath) => {
  return new Promise((resolve, reject) => {
    const token = wx.getStorageSync('userToken')
    wx.uploadFile({
      url: BASE_URL + '/upload/image',
      filePath,
      name: 'file',
      header: {
        'Authorization': token ? `Bearer ${token}` : ''
      },
      success: (res) => {
        let responseData
        try {
          responseData = JSON.parse(res.data)
        } catch (e) {
          wx.showToast({ title: '响应解析失败', icon: 'none' })
          reject(e)
          return
        }

        if (responseData.code === 0) {
          resolve(responseData)
        } else if (responseData.code === 401) {
          wx.removeStorageSync('userToken')
          wx.removeStorageSync('userInfo')
          wx.redirectTo({ url: '/pages/login/login' })
          reject(new Error(responseData.message || '登录已过期'))
        } else {
          wx.showToast({ title: responseData.message || '上传失败', icon: 'none' })
          reject(new Error(responseData.message || '上传失败'))
        }
      },
      fail: (err) => {
        wx.showToast({ title: '网络异常', icon: 'none' })
        reject(err)
      }
    })
  })
}

module.exports = {
  BASE_URL,
  ASSET_BASE_URL,
  request,
  api,
  uploadDiagnosisImage,
  getAssetUrl: (path) => {
    if (!path) return ''
    return /^https?:\/\//.test(path) ? path : ASSET_BASE_URL + path
  }
}
