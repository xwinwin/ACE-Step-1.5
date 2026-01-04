"""
Studio Bridge - Frontend JavaScript for ACE Studio WebBridge API

This module contains JavaScript code that runs in the browser to communicate
with ACE Studio WebBridge. Since Studio runs on the user's local machine
(localhost:21573), the API must be called from the frontend, not the backend.

Usage in Gradio:
    from studio_bridge import STUDIO_BRIDGE_JS, JS_CONNECT_STUDIO, ...
    
    with gr.Blocks(js=STUDIO_BRIDGE_JS) as demo:
        ...
        btn.click(fn=..., js=JS_CONNECT_STUDIO)
"""

# =============================================================================
# Main Studio Bridge Object
# =============================================================================

STUDIO_BRIDGE_JS = """
// Studio Bridge - Frontend JavaScript for ACE Studio WebBridge API
window.StudioBridge = {
    BASE_URL: 'https://localhost:21573',
    token: null,
    connected: false,

    // Connect to Studio with token
    async connect(token) {
        if (!token || !token.trim()) {
            return '❌ Please enter a token';
        }
        this.token = token.trim();
        try {
            const resp = await fetch(this.BASE_URL + '/bridge/token', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + this.token
                },
                body: JSON.stringify({ token: this.token })
            });
            if (resp.ok) {
                this.connected = true;
                return '✅ Connected to ACE Studio';
            } else {
                this.connected = false;
                return '❌ Connection failed: ' + resp.status;
            }
        } catch (e) {
            this.connected = false;
            if (e.message.includes('fetch')) {
                return '❌ Cannot reach Studio. Is it running? (Check localhost:21573)';
            }
            return '❌ Error: ' + e.message;
        }
    },

    // Get audio from Studio clipboard
    async getAudio() {
        if (!this.token) {
            return { error: '❌ Not connected. Please connect first.' };
        }
        try {
            // Get clipboard content
            const resp = await fetch(this.BASE_URL + '/bridge/clipboard', {
                method: 'GET',
                headers: { 'Authorization': 'Bearer ' + this.token }
            });
            if (!resp.ok) {
                return { error: '❌ Failed to get clipboard: ' + resp.status };
            }
            const data = await resp.json();
            if (!data.audio_url) {
                return { error: '❌ No audio in clipboard' };
            }
            // Return the audio URL for Gradio to fetch
            return { url: data.audio_url, message: '✅ Got audio from Studio' };
        } catch (e) {
            return { error: '❌ Error: ' + e.message };
        }
    },

    // Send audio to Studio
    async sendAudio(audioUrl, filename) {
        if (!this.token) {
            return '❌ Not connected. Please connect first.';
        }
        if (!audioUrl) {
            return '❌ No audio to send';
        }
        try {
            // Import audio to Studio
            const resp = await fetch(this.BASE_URL + '/bridge/audio/import', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + this.token
                },
                body: JSON.stringify({
                    src: audioUrl,
                    name: filename || 'ACEStep_Audio'
                })
            });
            if (!resp.ok) {
                return '❌ Failed to send: ' + resp.status;
            }
            const data = await resp.json();
            // Poll for import completion
            const taskId = data.taskId;
            if (taskId) {
                for (let i = 0; i < 30; i++) {
                    await new Promise(r => setTimeout(r, 500));
                    const statusResp = await fetch(this.BASE_URL + '/bridge/audio/import/' + taskId, {
                        headers: { 'Authorization': 'Bearer ' + this.token }
                    });
                    if (statusResp.ok) {
                        const status = await statusResp.json();
                        if (status.status === 'completed') {
                            return '✅ Audio sent to Studio';
                        } else if (status.status === 'failed') {
                            return '❌ Import failed: ' + (status.error || 'Unknown error');
                        }
                    }
                }
                return '⏳ Import started (taskId: ' + taskId + ')';
            }
            return '✅ Audio sent to Studio';
        } catch (e) {
            return '❌ Error: ' + e.message;
        }
    }
};
"""


# =============================================================================
# Gradio Event Handler JavaScript Functions
# =============================================================================

JS_CONNECT_STUDIO = """
async (token) => {
    const result = await window.StudioBridge.connect(token);
    return result;
}
"""

JS_GET_AUDIO_FROM_STUDIO = """
async () => {
    const result = await window.StudioBridge.getAudio();
    if (result.error) {
        return [null, result.error];
    }
    // Return the URL - Gradio will fetch it
    return [result.url, result.message];
}
"""

JS_SEND_AUDIO_TO_STUDIO = """
async (audioData) => {
    // audioData is the Gradio audio component value
    // It could be a URL, blob URL, or file path depending on context
    if (!audioData) {
        return '❌ No audio to send';
    }
    // Get the audio URL from Gradio component
    let audioUrl = audioData;
    if (typeof audioData === 'object' && audioData.url) {
        audioUrl = audioData.url;
    }
    const result = await window.StudioBridge.sendAudio(audioUrl, 'ACEStep_Generated.mp3');
    return result;
}
"""


# =============================================================================
# Exported Constants
# =============================================================================

__all__ = [
    'STUDIO_BRIDGE_JS',
    'JS_CONNECT_STUDIO',
    'JS_GET_AUDIO_FROM_STUDIO',
    'JS_SEND_AUDIO_TO_STUDIO',
]
