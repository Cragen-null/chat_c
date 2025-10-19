export class BellaCore {
  constructor() {
    this.videoElement = document.getElementById('video-bg');
  }

  playVideo(src, loop = false) {
    if (this.videoElement.getAttribute('src') === src) return; 
    this.videoElement.src = src;
    this.videoElement.loop = loop;
    this.videoElement.play().catch(err => console.error('再生エラー:', err));
  }

  playIdle() {
    this.playVideo('/static/video_resources/idle.mp4', true);
  }

  playListening() {
    this.playVideo('/static/video_resources/listening.mp4');
  }

  playTalking() {
    this.playVideo('/static/video_resources/talking.mp4');
  }

  static async getInstance() {
    // シングルトン
    if (!window._bellaInstance) {
      window._bellaInstance = new BellaCore();
    }
    return window._bellaInstance;
  }

  async think(text) {
    // 仮: バックエンドAPI呼び出し
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    return data.reply;
  }
}
