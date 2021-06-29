import * as T from '../../utils/index'//'../../utils/three'
import { registerGLTFLoader } from '../../utils/gltf-loader';
const fs = wx.getFileSystemManager()
wx.cloud.init()
// pages/home/home.js
Page({
  /**
   * 页面的初始数据
   */
  data: {
    usr_imgs:["img/filmtocat.png"],
    // usr_imgs:["cloud://aihuayin0125.6169-aihuayin0125-1301103558/MVS/back/0.jpg"],
    img_num:0,
    img_box_width:600,
    img_box_height:600,
    img_fetched:false,
    pic_button:'打开图片',
    progress:-1
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    //'cloud://aihuayin0125.6169-aihuayin0125-1301103558/MVS/back/points.ply'
    //初始化webgl
    // this.initWebGLCanvas();
    this.setData({
      progress:-1,
      usr_imgs:["img/filmtocat.png"]
    })
  },

  /**
   * 初始化Canvas对象
   */
  initWebGLCanvas:function(){
    //获取页面上的标签id为webgl的对象，从而获取到canvas对象
    const query = wx.createSelectorQuery();
    query.select('#webgl').node().exec((res) => {
      var canvas = res[0].node;
      this.THREE = T.createScopedThreejs(canvas);
      registerGLTFLoader(this.THREE);
      this.stage3d = canvas;
      //设置canvas的大小
      this.stage3d.width = this.data.img_box_width;
      this.stage3d.height = this.data.img_box_height;
      console.log("webgl : ",this.stage3d);
      //设置canvas的样式
      // this.stage3d.style = {};
      // this.stage3d.style.width = this.data.img_box_width;
      // this.stage3d.style.height = this.data.img_box_height;
      // console.log("who is this : ",this.data.usr_imgs[0]);
      this.initWebGLScene();
    });
  },
  initWebGLScene:function(){
    //创建摄像头
    var camera = new this.THREE.PerspectiveCamera(60, this.stage3d.width /this.stage3d.height , 0.1, 1000);
    this._camera = camera;
    //创建场景
    var scene = new this.THREE.Scene();
    scene.background = new this.THREE.Color( 0x000000 );
    this._scene = scene;
    wx.cloud.downloadFile({
      fileID:'cloud://aihuayin0125.6169-aihuayin0125-1301103558/MVS/back/points.json',
      success:(res)=>{
        console.log("点云路径：",res);
        wx.request({
          url: res.tempFilePath,
          header:{
            'content-type':'application/json'
          },
          success:(res)=>{
            this.setData({
              pointsCloud:res.data['loc'],
              pointsColor:res.data['color']
            })
            const geometry = new this.THREE.BoxGeometry( 10, 10, 10 );
            // for ( let i = 0; i < 200/*this.data.pointsCloud.length*/; i ++ ) {
            //   const object = new this.THREE.Mesh( geometry, new this.THREE.MeshLambertMaterial( { color: Math.random() * 0xffffff } ) );
            //   console.log("添加点",i)//," :",this.data.pointsCloud[i])
            //   object.position.x = this.data.pointsCloud[i][0];
            //   object.position.y = this.data.pointsCloud[i][1];
            //   object.position.z = this.data.pointsCloud[i][2];
            //   // object.position.x = Math.random() * 800 - 400;
            //   // object.position.y = Math.random() * 800 - 400;
            //   // object.position.z = Math.random() * 800 - 400;
            //   console.log("坐标： ",object.position);
            //   this._scene.add( object );
            // }
          }
        })
      },
      fail:(err)=>{
        console.log("点云文件下载失败");
      }
    })
    
    const geom = new this.THREE.BoxGeometry( 5, 5, 5 );
    const obje = new this.THREE.Points( geom, new this.THREE.PointsMaterial( { color: 0x0000ff } ));
    obje.position.x = 0;
    obje.position.y = 0;
    obje.position.z = 0;
    this._scene.add(obje);
    //创建渲染器
    this.renderer = new this.THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.stage3d.width, this.stage3d.height);
    console.log("开始渲染>>>>>>>>>>>>>>");
    this.renderer.render(this._scene, this._camera);
    console.log("渲染完成>>>>>>>>>>>>>>");

  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {
  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {
  },
  size_adaption:function(e){
    var width=e.detail.width;  //获取图片真实宽度  
    var height=e.detail.height;
    var ratio=height/width;   //图片的真实高宽比例  
    var box_ratio = this.data.img_box_height/this.data.img_box_width;
    if(ratio > box_ratio){
      var viewHeight=this.data.img_box_height,   
          viewWidth=viewHeight/ratio;   
    }else{
      var viewWidth=this.data.img_box_width,   
          viewHeight=viewWidth*ratio; 
    }
    if(this.data.usr_imgs.length>1 && this.data.usr_imgs.length<= 9){
      viewHeight = viewHeight/3;
      viewWidth = viewWidth/3;
    }else if(this.data.usr_imgs.length > 9){
      viewHeight = viewHeight/4;
      viewWidth = viewWidth/4;
    }
    else{ 
    }
    this.setData({  
        img_width:viewWidth,  
        img_height:viewHeight  
    })
  },
  fetch_img:function(){
    var that = this
    wx.showActionSheet({
      itemList: ['拍照','从相册选择'],
      success(res) {
        console.log(res.tapIndex)
        if(res.tapIndex==0){ 
          wx.chooseImage({
            count: 1,
            sizeType: ['compressed'],
            sourceType: ['camera'],
            success: (res)=> {
              that.setData({
                usr_img:res.tempFilePaths,
                img_fetched:true
              })//res.tempFilePaths[0] 是图片
             },
          })
        } else if(res.tapIndex==1){
          wx.chooseImage({
            count: 8,
            sizeType: ['compressed'],
            sourceType: ['album'],
            success: (res)=> {
              console.log(res.tempFilePaths);
              that.setData({
                usr_imgs:res.tempFilePaths,
                img_num:res.tempFilePaths.length, 
                img_fetched:true
              })
              console.log("图片数量 ： ", that.data.img_num);
            },
          })
        }
      }
    })
    },
  preview_img:function(e){
    console.log(e);
    var cur_img = e.currentTarget.dataset.url;
    var all_imgs = this.data.usr_imgs;
    wx.previewImage({
      current: cur_img,
      urls: all_imgs,
      success: function(res) {},
      fail: function(res) {},
      complete: function(res) {},
    })
  },
  reconstruction:function(){
    var imgs = this.data.usr_imgs;
    var count = 0;
    var wait = ['Everything comes to him who waits', '我们学会了追赶时间，同样要学会耐心等待', '慢工出细活啊', '心急可吃不了热豆腐！', 'O(∩_∩)O', '生活从来不缺少惊喜，但你要耐心等待', '有时候，最大的果实，需要耐心等待', '余生很长，请耐心等待，别太早离开', '不经历等待，怎能见彩虹？', '最极致的坚持，是耐心等待', '无论何人，若是失去耐心，便失去了灵魂', '耐心等待，也是一种善良', '善于等待的人，一切都会及时来到', '我话还没说完呢','不经历等待，怎能见彩虹？','耐心，是一切聪明才智的基础', '耐心和恒心总会得到报酬的','有耐心的人，能得到他所期望的','谁没有耐心，谁就没有智慧','哪里有天才，我是把别人喝咖啡的工夫都用在三维重建上','要看日出必须守到拂晓','耐心是希望的艺术','不经历等待，怎能见彩虹？','忍耐是痛的，但是它的结果是甜蜜的','耐心之树，结黄金之果','事业常成于坚忍，毁于急躁','我们所需要的，或许只是耐心等待','不经历等待，怎能见彩虹？','骆驼走得慢，但终能走到目的地',]
    wx.showToast({
      title: '重建，已然开始！',
      icon: 'success',
      duration: 900
    })
    setTimeout(()=>{
      wx.showToast({
        title: '客官稍安勿躁',
        icon: 'none',
        duration: 2000
      })
    },5000)
    setTimeout(()=>{
      wx.showToast({
        title: '重建即将完成',
        icon: 'none',
        duration: 2000
      })
    },10000)
    for(let i=0;i<wait.length;i++){
      setTimeout(()=>{
        wx.showToast({
          title: wait[i],
          icon: 'none',
          duration: 2000
        })
      },9000*(i+2))
    }
    // 首先将文件上传到云服务器
    for(let i=0;i<imgs.length;i++){
      wx.cloud.uploadFile({
        cloudPath: "MVS/go/"+i+".jpg",
        filePath: imgs[i], // 文件路径
        success: (res) => {
          // get resource ID
          console.log(res.fileID);
          count += 1;
          if(count == imgs.length){
            //向fastmvs服务器发送post请求
            console.log("向fastmvs服务器发送请求");
            wx.request({
              url: 'http://127.0.0.1:5000/fastmvs/'+imgs.length,
              method: "POST",//指定请求方式，默认get
              data: {},
              header: {
                //默认值'Content-Type': 'application/json'
                'content-type': 'application/x-www-form-urlencoded' //post
              },
              success: function (res) {
                console.log(res.data)
              },
              fail: function(res){
                console.log(res);
              }
            });
          }
        },
        fail: (err) => {
          // handle error
          console.log("err : ",err);
        }
      })
    }
    var _this = this
    console.log("this : ",_this)
    for(let i=1;i<=600;i++){
      setTimeout(()=>{
        console.log("current progress : ",_this.data.progress)
        if(_this.data.progress == _this.data.usr_imgs.length - 1){
          return;
        }
        wx.request({
          url: 'http://127.0.0.1:5000/ask_and_report',
          method: "POST",//指定请求方式，默认get
          data: {},
          header: {
            //默认值'Content-Type': 'application/json'
            'content-type': 'application/x-www-form-urlencoded' //post
          },
          success: function (res) {
            if(res.data > _this.data.progress){
              if(res.data == 0){
                wx.showToast({
                  title: '看，彩虹！',
                  icon:'none',
                  duration: 1200
                })
                wx.cloud.downloadFile({
                  fileID:'cloud://aihuayin0125.6169-aihuayin0125-1301103558/MVS/back/points.json',
                  success:(res)=>{
                    console.log("点云路径：",res);
                    wx.request({
                      url: res.tempFilePath,
                      header:{
                        'content-type':'application/json'
                      },
                      success:(res)=>{
                        this.setData({
                          pointsCloud:res.data['loc'],
                          pointsColor:res.data['color']
                        })
                      }
                    })
                  },
                  fail:(err)=>{
                    console.log("点云文件下载失败");
                  }
                })
              }
              //首先更新progress的值
              _this.data.progress = res.data;
              console.log("current progress : ", _this.data.progress);
              //然后加载深度重建结果
              _this.data.usr_imgs.splice(2 * _this.data.progress+1,0,"cloud://aihuayin0125.6169-aihuayin0125-1301103558/MVS/back/"+String(_this.data.progress)+".jpg");
              _this.setData({
                usr_imgs:_this.data.usr_imgs
              });
              console.log("usr imgs : ",_this.data.usr_imgs);
            }
          }
        });
      },2000*i);
    }
  },

  // 获取保存图片的权限
  save_pic_auth: function(e) {
    let _this = this
    wx.showActionSheet({
        itemList: ['保存到相册'],
        success(res) {
            let url = e.currentTarget.dataset.url;
            wx.getSetting({
                success: (res) => {
                    if (!res.authSetting['scope.writePhotosAlbum']) {
                        wx.authorize({
                            scope: 'scope.writePhotosAlbum',
                            success: () => {
                                // 同意授权
                                _this.save_pic(url);
                            },
                            fail: (res) => {
                                console.log(res);
                                wx.showModal({
                                    title: '保存失败',
                                    content: '请开启访问手机相册权限',
                                    success(res) {
                                        wx.openSetting()
                                    }
                                })
                            }
                        })
                    } else {
                        // 已经授权了
                        _this.save_pic(url);
                    }
                },
                fail: (res) => {
                    console.log(res);
                }
            })   
        },
        fail(res) {
            console.log(res.errMsg)
        }
    })
  },
  // 保存图片
  save_pic:function(url){
    wx.getImageInfo({
      src: url,
      success: (res) => {
          let path = res.path;
          wx.saveImageToPhotosAlbum({
              filePath: path,
              success: (res) => {
                  console.log(res);
                  wx.showToast({
                      title: '保存成功',
                  })
              },
              fail: (res) => {
                  console.log(res);
              }
          })
      },
      fail: (res) => {
          console.log(res);
      }
    })
  },
  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})