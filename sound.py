import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import time
import colorsys

# Enhanced vertex shader with normal calculation for better lighting
VERTEX_SHADER = """
#version 330
in vec2 position;
out vec3 fragNormal;
out vec3 fragPosition;
out float waveHeight;

uniform float time;
uniform float frequency;
uniform float amplitude;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

vec3 calculateNormal(vec2 pos, float freq, float t, float amp) {
    // Calculate derivatives for normal computation
    float dx = freq * 2.0 * pos.x * cos(freq * (pos.x * pos.x + pos.y * pos.y) - t)
              - freq * sin(freq * (pos.x + pos.y) - t/2.0);
    float dy = freq * 2.0 * pos.y * cos(freq * (pos.x * pos.x + pos.y * pos.y) - t)
              - freq * sin(freq * (pos.x + pos.y) - t/2.0);
    
    // Normal = cross product of tangent vectors
    return normalize(vec3(-dx * amp, 1.0, -dy * amp));
}

void main(){
    // Compute wave height with improved formula
    float z = amplitude * sin(frequency * (position.x * position.x + position.y * position.y) - time)
            * cos(frequency * (position.x + position.y) - time/2.0);
    
    // Calculate normal for lighting
    fragNormal = calculateNormal(position, frequency, time, amplitude);
    
    // Transform position
    vec4 worldPos = model * vec4(position.x, z, position.y, 1.0);
    fragPosition = worldPos.xyz;
    waveHeight = z;
    
    // Final position
    gl_Position = projection * view * worldPos;
}
"""

# Enhanced fragment shader with dynamic color based on wave height and Phong lighting
FRAGMENT_SHADER = """
#version 330
in vec3 fragNormal;
in vec3 fragPosition;
in float waveHeight;
out vec4 fragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 baseColor;
uniform float colorShift;
uniform float time;

void main(){
    // Normalized height for color calculation
    float normHeight = (waveHeight + 1.0) / 2.0;
    
    // Dynamic wave color based on height and time
    vec3 waveColor = baseColor;
    waveColor.r = baseColor.r + normHeight * colorShift * sin(time/5.0);
    waveColor.g = baseColor.g + normHeight * colorShift * cos(time/7.0);
    waveColor.b = baseColor.b + normHeight * colorShift;
    
    // Ambient light
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * waveColor;
    
    // Diffuse light
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPosition);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * waveColor;
    
    // Specular light
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - fragPosition);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
    
    // Final color
    vec3 result = ambient + diffuse + specular;
    fragColor = vec4(result, 1.0);
}
"""

def create_rotation_matrix(angle, axis):
    """Creates a rotation matrix for a given angle and axis."""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    
    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c), 0],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b), 0],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def perspective(fovy, aspect, near, far):
    """Creates a perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fovy) / 2)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj

def lookAt(eye, center, up):
    """Creates a view matrix using the lookAt convention."""
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = -eye[0]
    T[1, 3] = -eye[1]
    T[2, 3] = -eye[2]
    return M @ T

class EnhancedWaveVisualizer:
    def __init__(self, grid_size=150, width=1024, height=768):
        # Display settings
        self.width = width
        self.height = height
        
        # Audio parameters
        self.SAMPLE_RATE = 44100
        self.current_frequency = 440.0  # Default frequency in Hz (A4)
        self.volume = 0.3
        self.harmonics = [(1.0, 1.0), (1.5, 0.5), (2.0, 0.25), (3.0, 0.125)]  # (frequency multiplier, amplitude)

        # Wave parameters
        self.amplitude = 0.6
        self.wave_speed = 1.0
        self.grid_size = grid_size
        self.color_shift = 0.3
        self.base_color = np.array([0.2, 0.7, 1.0], dtype=np.float32)
        
        # Camera parameters
        self.camera_position = np.array([0.0, 3.0, 5.0], dtype=np.float32)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.camera_rotation_speed = 0.3
        self.auto_rotate = True
        
        # Light parameters
        self.light_position = np.array([2.0, 4.0, 2.0], dtype=np.float32)
        self.light_moving = True
        
        # Timing
        self.start_time = time.time()
        self.last_fps_update = 0
        self.frame_count = 0
        self.fps = 0

        # Build the grid for visualization
        self.build_grid()

        # Set font for overlay
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 18)
        
        # Initialize audio
        pygame.mixer.init(frequency=self.SAMPLE_RATE, size=-16, channels=1)
        self.sound_playing = False
        self.sound_channel = None
        
        # Flags for interactivity
        self.wireframe_mode = False
        self.show_info = True
        
        # Initialize OpenGL after setting display
        self.setup_display()
        self.init_opengl()
        self.update_sound()

    def setup_display(self):
        """Set up the Pygame display."""
        self.display = (self.width, self.height)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Enhanced Wave Visualizer")

    def build_grid(self):
        """Create the grid geometry data."""
        # Create vertices
        xs = np.linspace(-np.pi, np.pi, self.grid_size, dtype=np.float32)
        ys = np.linspace(-np.pi, np.pi, self.grid_size, dtype=np.float32)
        self.vertices = []
        for y in ys:
            for x in xs:
                self.vertices.append([x, y])
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vertex_count = len(self.vertices)
        
        # Create indices for triangle drawing
        self.indices = []
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size - 1):
                top_left = i * self.grid_size + j
                top_right = top_left + 1
                bottom_left = (i + 1) * self.grid_size + j
                bottom_right = bottom_left + 1
                # Two triangles per quad
                self.indices.extend([top_left, bottom_left, top_right])
                self.indices.extend([top_right, bottom_left, bottom_right])
        self.indices = np.array(self.indices, dtype=np.uint32)
        self.index_count = len(self.indices)

    def init_opengl(self):
        """Initialize OpenGL settings and resources."""
        # Compile and link shaders
        self.shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        
        # Create VAO, VBO, and EBO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        
        pos_loc = glGetAttribLocation(self.shader, "position")
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
        
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # Initialize matrices
        self.update_matrices()

    def update_matrices(self):
        """Update model, view, and projection matrices."""
        self.model = np.eye(4, dtype=np.float32)
        self.view = lookAt(self.camera_position, self.camera_target, self.camera_up)
        self.projection = perspective(45, self.width / self.height, 0.1, 100)

    def update_camera(self, delta_time):
        """Update camera position for auto-rotation."""
        if self.auto_rotate:
            # Rotate camera around the target
            rotation = create_rotation_matrix(delta_time * self.camera_rotation_speed, np.array([0, 1, 0]))
            position = np.append(self.camera_position - self.camera_target, 1)
            rotated_position = rotation @ position
            self.camera_position = self.camera_target + rotated_position[:3]
            self.update_matrices()

    def update_light(self, current_time):
        """Update light position for dynamic lighting."""
        if self.light_moving:
            radius = 3.0
            self.light_position[0] = radius * np.sin(current_time * 0.5)
            self.light_position[2] = radius * np.cos(current_time * 0.5)
            self.light_position[1] = 3.0 + np.sin(current_time * 0.3)

    def generate_sound(self, freq, duration=2.0):
        """Generate a sound sample with harmonics."""
        t_vals = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        wave = np.zeros_like(t_vals)
        
        # Add multiple harmonics
        for harmonic, amplitude in self.harmonics:
            wave += amplitude * self.volume * np.sin(2 * np.pi * freq * harmonic * t_vals)
        
        # Apply a simple envelope to avoid clicks
        envelope = np.ones_like(t_vals)
        attack_samples = int(0.01 * self.SAMPLE_RATE)  # 10ms attack
        decay_samples = int(0.05 * self.SAMPLE_RATE)   # 50ms decay
        
        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        if decay_samples > 0:
            decay_start = len(envelope) - decay_samples
            envelope[decay_start:] = np.linspace(1, 0, decay_samples)
        
        wave = wave * envelope
        
        # Convert to 16-bit audio
        wave = (wave * 32767).astype(np.int16)
        
        # Fix the audio channel issue - ensure proper shape for mono audio
        # The error was because we need a 1D array for mono, not a 2D array
        return pygame.mixer.Sound(buffer=wave)

    def update_sound(self):
        """Update the looping sound to match the current frequency."""
        try:
            if self.sound_playing and self.sound_channel:
                self.sound_channel.stop()
            
            sound = self.generate_sound(self.current_frequency)
            self.sound_channel = sound.play(-1)  # Loop indefinitely
            self.sound_playing = True
        except Exception as e:
            print(f"Sound error: {e}")
            # Continue without sound if there's an error
            self.sound_playing = False

    def handle_keyboard_input(self, delta_time):
        """Process keyboard input for interactive controls."""
        keys = pygame.key.get_pressed()
        
        # Frequency controls
        freq_change = 0
        if keys[K_UP]:
            freq_change = 10
        elif keys[K_DOWN]:
            freq_change = -10
        
        if freq_change != 0:
            self.current_frequency = max(20, self.current_frequency + freq_change)
            if self.sound_playing:
                self.update_sound()
            
        # Amplitude controls
        if keys[K_PAGEUP]:
            self.amplitude = min(1.5, self.amplitude + 0.02)
        elif keys[K_PAGEDOWN]:
            self.amplitude = max(0.1, self.amplitude - 0.02)
            
        # Camera controls
        speed = 2.0 * delta_time
        if keys[K_w]:
            self.camera_position[2] -= speed
        if keys[K_s]:
            self.camera_position[2] += speed
        if keys[K_a]:
            self.camera_position[0] -= speed
        if keys[K_d]:
            self.camera_position[0] += speed
        if keys[K_q]:
            self.camera_position[1] += speed
        if keys[K_z]:
            self.camera_position[1] -= speed
            
        if keys[K_r]:
            # Reset camera
            self.camera_position = np.array([0.0, 3.0, 5.0], dtype=np.float32)
            self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
        # Update matrices if camera changed
        if keys[K_w] or keys[K_s] or keys[K_a] or keys[K_d] or keys[K_q] or keys[K_z] or keys[K_r]:
            self.update_matrices()

    def render_frame(self, current_time, delta_time):
        """Render a single frame."""
        # Update simulation
        self.update_camera(delta_time)
        self.update_light(current_time)
        
        # Clear the frame
        glClearColor(0.05, 0.05, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Use shader program
        glUseProgram(self.shader)
        
        # Update uniforms
        time_loc = glGetUniformLocation(self.shader, "time")
        glUniform1f(time_loc, current_time * self.wave_speed)
        
        freq_loc = glGetUniformLocation(self.shader, "frequency")
        wave_freq = (self.current_frequency / 440.0) * 1.5
        glUniform1f(freq_loc, wave_freq)
        
        amp_loc = glGetUniformLocation(self.shader, "amplitude")
        glUniform1f(amp_loc, self.amplitude)
        
        # Set matrices
        model_loc = glGetUniformLocation(self.shader, "model")
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, self.model)
        
        view_loc = glGetUniformLocation(self.shader, "view")
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, self.view)
        
        proj_loc = glGetUniformLocation(self.shader, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, self.projection)
        
        # Set lighting
        light_pos_loc = glGetUniformLocation(self.shader, "lightPos")
        glUniform3fv(light_pos_loc, 1, self.light_position)
        
        view_pos_loc = glGetUniformLocation(self.shader, "viewPos")
        glUniform3fv(view_pos_loc, 1, self.camera_position)
        
        # Set color parameters
        base_color_loc = glGetUniformLocation(self.shader, "baseColor")
        glUniform3fv(base_color_loc, 1, self.base_color)
        
        color_shift_loc = glGetUniformLocation(self.shader, "colorShift")
        glUniform1f(color_shift_loc, self.color_shift)
        
        # Draw mesh
        glBindVertexArray(self.VAO)
        
        if self.wireframe_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        # Render overlay information
        if self.show_info:
            self.render_overlay()

    def render_overlay(self):
        """Render text overlay with information."""
        # Switch to 2D rendering for text
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        
        # Draw text overlay
        self.draw_text(10, 10, f"FPS: {self.fps:.1f}")
        self.draw_text(10, 30, f"Frequency: {self.current_frequency:.1f} Hz")
        self.draw_text(10, 50, f"Amplitude: {self.amplitude:.2f}")
        self.draw_text(10, 70, f"Wireframe: {'On' if self.wireframe_mode else 'Off'}")
        self.draw_text(10, 90, f"Auto-rotate: {'On' if self.auto_rotate else 'Off'}")
        
        # Controls help
        y_pos = self.height - 130
        self.draw_text(10, y_pos, "Controls:")
        self.draw_text(10, y_pos + 20, "↑/↓: Adjust frequency")
        self.draw_text(10, y_pos + 40, "PgUp/PgDn: Adjust amplitude")
        self.draw_text(10, y_pos + 60, "W/A/S/D/Q/Z: Move camera")
        self.draw_text(10, y_pos + 80, "R: Reset camera, F: Toggle wireframe, Space: Toggle rotation")
        self.draw_text(10, y_pos + 100, "I: Toggle info, C: Cycle colors, Esc: Quit")
        
        # Restore 3D rendering
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def draw_text(self, x, y, text, color=(255, 255, 255)):
        """Draw text on the screen."""
        try:
            text_surface = self.font.render(text, True, color)
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            width, height = text_surface.get_size()
            
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glRasterPos2d(x, y)
            glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            glDisable(GL_BLEND)
        except Exception as e:
            print(f"Text rendering error: {e}")

    def update_fps(self, current_time):
        """Update FPS counter."""
        self.frame_count += 1
        if current_time - self.last_fps_update >= 1.0:  # Update every second
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time

    def run(self):
        """Main application loop."""
        clock = pygame.time.Clock()
        running = True
        last_time = time.time()
        
        # Main loop
        while running:
            current_time = time.time() - self.start_time
            delta_time = time.time() - last_time
            last_time = time.time()
            
            # Process events
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_f:
                        self.wireframe_mode = not self.wireframe_mode
                    elif event.key == K_i:
                        self.show_info = not self.show_info
                    elif event.key == K_SPACE:
                        self.auto_rotate = not self.auto_rotate
                    elif event.key == K_l:
                        self.light_moving = not self.light_moving
                    elif event.key == K_c:
                        # Cycle through color presets
                        colors = [
                            [0.2, 0.7, 1.0],  # Blue
                            [0.8, 0.2, 0.8],  # Purple
                            [0.1, 0.8, 0.3],  # Green
                            [0.9, 0.6, 0.1]   # Orange
                        ]
                        current_idx = 0
                        for i, color in enumerate(colors):
                            if np.allclose(self.base_color, color):
                                current_idx = i
                                break
                        self.base_color = np.array(colors[(current_idx + 1) % len(colors)], dtype=np.float32)
                    elif event.key == K_m:
                        # Toggle sound
                        if self.sound_playing:
                            if self.sound_channel:
                                self.sound_channel.stop()
                            self.sound_playing = False
                        else:
                            self.update_sound()
            
            # Process continuous keyboard input
            self.handle_keyboard_input(delta_time)
            
            # Render frame
            self.render_frame(current_time, delta_time)
            
            # Update FPS counter
            self.update_fps(current_time)
            
            # Display the rendered frame
            pygame.display.flip()
            
            # Cap the frame rate
            clock.tick(60)
        
        # Clean up
        if self.sound_playing and self.sound_channel:
            self.sound_channel.stop()
        pygame.quit()

if __name__ == "__main__":
    pygame.init()
    # Start with a smaller grid size for better performance
    visualizer = EnhancedWaveVisualizer(grid_size=100, width=1024, height=768)
    visualizer.run()
