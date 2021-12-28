#define NOMINMAX
#include<iostream>
#include<fstream>
#include<iomanip>
#include<cstring>
#include<atomic>
#include<chrono>
#include<memory>
#include<vector>
#include<thread>
#include<string>
#include<mutex>
#include<cmath>
#include<glad/glad.h>
#include<GLFW/glfw3.h>

template<typename T>
struct Vector2 {
	Vector2() = default;
	template<typename U>
	Vector2(const U& v) :x((T)v), y((T)v) {}
	template<typename U>
	Vector2(const U& x, const U& y) : x((T)x), y((T)y) {}
	T x{}, y{};
};

template<typename T>
struct Vector3 {
	Vector3() = default;
	template<typename U>
	Vector3(const U& v) :x((T)v), y((T)v), z((T)v) {}
	template<typename U>
	Vector3(const U& x, const U& y, const U& z) : x((T)x), y((T)y), z((T)z) {}
	T x{}, y{}, z{};
};

using u8vec2 = Vector2<uint8_t>;
using u8vec3 = Vector3<uint8_t>;
using ivec2 = Vector2<int>;
using ivec3 = Vector3<int>;
using fvec2 = Vector2<float>;
using fvec3 = Vector3<float>;
using dvec2 = Vector2<double>;
using dvec3 = Vector3<double>;

template<typename T, typename U>
U lerp(T t, U a, T b) {
	return ((T)1 - t) * a + t * b;
}
template<typename T>
T mul2(T fp) {
	return fp * 2;
}
template<typename T>
T sqr(T fp) {
	return fp * fp;
}

inline fvec3 colormap(float f01) {
	if (f01 < 1.0f / 3)return { lerp(f01 * 3,0.0f,0.0f),lerp(f01 * 3,0.0f,0.0f) ,lerp(f01 * 3,.2f,0.8f) };
	f01 -= 1.0f / 3;
	if (f01 < 1.0f / 3)return { lerp(f01 * 3,0.0f,0.4f),lerp(f01 * 3,0.0f,0.6f) ,lerp(f01 * 3,0.8f,1.0f) };
	f01 -= 1.0f / 3;
	return { lerp(f01 * 3,0.4f,1.0f),lerp(f01 * 3,0.6f,1.0f) ,lerp(f01 * 3,1.0f,1.0f) };
}

struct Image {
	Image(ivec2 size) : size(size), colors(new fvec3[size.x * size.y]) {}
	fvec3& operator[](ivec2 p) {
		return colors[p.x + p.y * size.x];
	}
	void clear() {
		for (int y = 0; y < size.y; y++)
			for (int x = 0; x < size.x; x++)
				colors[x + y * size.x] = 0;
	}
	void write_to_file(std::string filename) {
		filename = filename + ".tga";
		std::ofstream file(filename, std::ios::binary);
		printf("\nwrite result image to:%s\n", filename.c_str());
		u8vec3* buf = new u8vec3[size.x * size.y];
		for (int y = 0; y < size.y; y++)
			for (int x = 0; x < size.x; x++) {
				fvec3 rgb = colors[x + y * size.x];
				std::swap(rgb.x, rgb.z);
				rgb.x = std::max(std::min(rgb.x, 1.0f), 0.0f);
				rgb.y = std::max(std::min(rgb.y, 1.0f), 0.0f);
				rgb.z = std::max(std::min(rgb.z, 1.0f), 0.0f);
				buf[x + (size.y - 1 - y) * size.x] = u8vec3(rgb.x * 255.99f, rgb.y * 255.99f, rgb.z * 255.99f);
			}
		uint8_t header[18] = { 0,0,2,0,0,0,0,0,0,0,0,0,
			(uint8_t)(size.x % 256), (uint8_t)(size.x / 256),
			(uint8_t)(size.y % 256), (uint8_t)(size.y / 256),
			(uint8_t)(3 * 8), 0x20 };
		file.write((const char*)header, 18);
		file.write((const char*)buf, (size_t)size.x * size.y * 3);
		delete[] buf;
		file.close();
	}
	ivec2 size;
	std::shared_ptr<fvec3[]> colors;
};

class Viewer {
public:
	Viewer(const Viewer&) = delete;
	const Viewer& operator=(const Viewer&) = delete;
	Viewer(Image image) :image(image) {
		glfwInit();
		glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE);
		GLFWmonitor* monitor = glfwGetPrimaryMonitor();
		const GLFWvidmode* mode = glfwGetVideoMode(monitor);
		if (mode->width != image.size.x || mode->height != image.size.y)monitor = nullptr;
		window = glfwCreateWindow(image.size.x, image.size.y, "Viewer", monitor, nullptr);
		glfwMakeContextCurrent(window);
		glfwSwapInterval(false);
		gladLoadGL();

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);
		glGenBuffers(1, &VBO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		float vertices[] = {
		-1.0f, 1.0f,
		-1.0f,-1.0f,
		 1.0f,-1.0f,
		-1.0f, 1.0f,
		 1.0f, 1.0f,
		 1.0f,-1.0f
		};
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void*)0);
		glEnableVertexAttribArray(0);

		program = glCreateProgram();
		GLuint vs = glCreateShader(GL_VERTEX_SHADER);
		GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
		const char* vsSource =
			"#version 450 core\n"
			"layout(location = 0)in vec2 aPos;"
			"out vec2 coord;"
			"void main(){"
			"gl_Position = vec4(aPos,0.0,1.0);"
			"coord = aPos * 0.5 + vec2(0.5);"
			"}";
		const char* fsSource =
			"#version 450 core\n"
			"out vec4 FragColor;"
			"layout(binding = 0)uniform sampler2D imageTexture;"
			"in vec2 coord;"
			"void main(){"
			"FragColor = texture(imageTexture,coord);"
			"}";
		glShaderSource(vs, 1, &vsSource, NULL);
		glShaderSource(fs, 1, &fsSource, NULL);
		glCompileShader(vs);
		glCompileShader(fs);
		glAttachShader(program, vs);
		glAttachShader(program, fs);
		glLinkProgram(program);
		glUseProgram(program);
		glDeleteShader(vs);
		glDeleteShader(fs);

		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glActiveTexture(GL_TEXTURE0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, image.size.x, image.size.y, 0, GL_RGB, GL_FLOAT, NULL);
	};
	~Viewer() {
		glDeleteProgram(program);
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		glDeleteBuffers(1, &texture);
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void render() {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, image.size.x, image.size.y, 0, GL_RGB, GL_FLOAT, image.colors.get());
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glFlush();

		glfwPollEvents();
	}
	bool should_close() const {
		if (glfwWindowShouldClose(window) || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			return true;
		else
			return false;
	}

	Image image;
	GLFWwindow* window = nullptr;
private:
	GLuint program = 0;
	GLuint VAO = 0;
	GLuint VBO = 0;
	GLuint texture = 0;
};

template<typename... Args>
void print(Args... args) {
	int expand[] = { 0,(std::cout << args,0)... };
	std::cout << std::endl;
}

template<typename T>
auto operator<<(std::ostream& os, const T& anything)
->decltype(anything.to_string(), std::declval<std::ostream&>()) {
	os << anything.to_string();
	return os;
}

template<typename F>
void parallel_for(ivec2 size, F f) {
	static std::vector<std::thread> threads(std::thread::hardware_concurrency());
	std::atomic<int> task_counter{};

	for (auto& t : threads)
		t = std::thread([&]() {
		int y = task_counter++;
		while (y < size.y) {
			for (int x = 0; x < size.x; x++)
				f({ x,y });
			y = task_counter++;
		} });
	for (auto& t : threads)
		t.join();
}

class Timer {
public:
	using Clock = std::chrono::steady_clock;
	Timer() {
		t0 = Clock::now();
	}
	double duration(double clamp = 1e+20) const {
		return std::min(std::chrono::duration<double>(Clock::now() - t0).count(), clamp);
	}
	void reset() {
		t0 = Clock::now();
	}
private:
	Clock::time_point t0{};
};

template<int nu32>
struct uinteger {
	constexpr uinteger() = default;
	constexpr uinteger(uint64_t val) {
		bits[0] = uint32_t(val);
		bits[1] = uint32_t(val >> 32);
	}
	constexpr const uinteger& operator+=(uinteger it) {
		for (int i = 0; i < nu32; i++)
			set_uint64((uint64_t)bits[i] + (uint64_t)it.bits[i], i);
		return *this;
	}
	constexpr const uinteger& operator-=(uinteger it) {
		for (int i = 0; i < nu32; i++) {
			if (bits[i] >= it.bits[i])
				bits[i] -= it.bits[i];
			else
				set_neg_uint32(it.bits[i] - bits[i], i);
		}
		return *this;
	}
	constexpr const uinteger& operator*=(uinteger it) {
		uinteger tmp = *this;
		*this = {};
		for (int i = 0; i < nu32; i++)
			for (int j = 0; j < nu32; j++)
				set_uint64((uint64_t)bits[i + j] + (uint64_t)tmp.bits[i] * (uint64_t)it.bits[j], i + j);

		return *this;
	}

	friend uinteger operator+(uinteger lhs, const uinteger& rhs) {
		return lhs += rhs;
	}
	friend uinteger operator-(uinteger lhs, const uinteger& rhs) {
		return lhs -= rhs;
	}
	friend uinteger operator*(uinteger lhs, const uinteger& rhs) {
		return lhs *= rhs;
	}

	constexpr void set_uint64(uint64_t val, int pos) {
		while (val != 0 && pos < nu32) {
			bits[pos] = (uint32_t)val;
			if (++pos < nu32)
				val = (val >> 32) + (uint64_t)bits[pos];
		}
	}
	constexpr void set_neg_uint32(uint32_t val, int pos) {
		while (pos < nu32) {
			bits[pos] = (uint32_t)0xffffffff - (val - 1);
			if (++pos < nu32)
				if (bits[pos] > 0) {
					bits[pos]--;
					break;
				}
			val = 1;
		}
	}
	template<typename T> constexpr
		T& as(int index) {
		return *((T*)bits + index);
	}
	template<typename T> constexpr
		const T& as(int index) const {
		return *((T*)bits + index);
	}

	constexpr uint32_t& operator[](int i) {
		return as<uint32_t>(i);
	}
	constexpr const uint32_t& operator[](int i) const {
		return as<uint32_t>(i);
	}

	template<int bit, bool value>
	constexpr void set() {
		constexpr int begin_u32 = bit / 32;
		if (value)
			bits[begin_u32] |= 1u << (bit - 32 * begin_u32);
		else
			bits[begin_u32] &= ~(1u << (bit - 32 * begin_u32));
	}
	template<bool value>
	void set(int bit) {
		int begin_u32 = bit / 32;
		if (value)
			bits[begin_u32] |= 1u << (bit - 32 * begin_u32);
		else
			bits[begin_u32] &= ~(1u << (bit - 32 * begin_u32));
	}
	template<int bit>
	constexpr bool get() {
		constexpr int begin_u32 = bit / 32;
		return bits[begin_u32] & (1u << (bit - 32 * begin_u32));
	}

	constexpr const uinteger& operator>>=(unsigned int b) {
		int bnu = b / 32;
		int bof = b - bnu * 32;

		uint32_t carry = 0;
		for (int i = 0; i < nu32 - bnu; i++) {
			carry = bits[i + bnu] << (32 - bof);
			bits[i] = bits[i + bnu] >> bof;
			if (bof && i != 0)bits[i - 1] |= carry;
		}

		for (int i = std::max(nu32 - bnu, 0); i < nu32; i++)
			bits[i] = 0;

		return *this;
	}
	constexpr const uinteger& operator<<=(unsigned int b) {
		int bnu = b / 32;
		int bof = b - bnu * 32;

		uint32_t carry = 0;
		for (int i = nu32 - 1; i >= bnu; i--) {
			carry = bits[i - bnu] >> (32 - bof);
			bits[i] = bits[i - bnu] << bof;
			if (bof && i != nu32 - 1)bits[i + 1] |= carry;
		}

		for (int i = std::min(bnu - 1, nu32 - 1); i >= 0; i--)
			bits[i] = 0;

		return *this;
	}

	constexpr uinteger operator>>(unsigned int bits) const {
		uinteger tmp = *this;
		return tmp >>= bits;
	}
	constexpr uinteger operator<<(unsigned int bits) const {
		uinteger tmp = *this;
		return tmp <<= bits;
	}

	constexpr bool operator!=(const uinteger& it) const {
		for (int i = nu32 - 1; i >= 0; i--)
			if (bits[i] != it.bits[i])return true;
		return false;
	}
	constexpr bool operator==(const uinteger& it) const {
		return !(*this != it);
	}

	friend  constexpr int compare(const uinteger& lhs, const uinteger& rhs) {
		for (int i = nu32 - 1; i >= 0; i--) {
			if (lhs.bits[i] > rhs.bits[i])return 1;
			if (lhs.bits[i] < rhs.bits[i])return -1;
		}
		return 0;
	}

	constexpr int bit_scan_reverse()const {
		for (int i = nu32 - 1; i >= 0; i--) {
			if (bits[i]) {
				uint32_t bit = bits[i];
				for (int j = 31; j >= 0; j--)
					if (1u && (bit >> j))
						return j + i * 32;
			}
		}
		return -1;
	}

	uint32_t bits[nu32]{ 0 };
};

class float64 {
public:
	float64() = default;
	float64(double value) {
		*this = value;
	}
	void operator=(double value) {
		memcpy(&bits, &value, sizeof(uint64_t));
	}
	operator double() const {
		return (sign() ? -1 : 1) *
			(double)std::pow(2ull, exponent() - exponent_offset) *
			(1ull + (double)mantissa() / (1ull << mantissa_bits));
	}
	uint64_t sign() const {
		return bits >> exponent_bits >> mantissa_bits;
	}
	int64_t exponent() const {
		return int64_t((bits & exponent_mask) >> mantissa_bits);
	}
	uint64_t mantissa() const {
		return bits & mantissa_mask;
	}
	uint64_t bits;
	static const uint64_t sign_bits = 1;
	static const uint64_t exponent_bits = 11;
	static const uint64_t mantissa_bits = 52;
	static const int64_t exponent_offset = (1 << (exponent_bits - 1)) - 1;
	static const uint64_t sign_mask = 0b1000000000000000000000000000000000000000000000000000000000000000;
	static const uint64_t exponent_mask = 0b0111111111110000000000000000000000000000000000000000000000000000;
	static const uint64_t mantissa_mask = 0b0000000000001111111111111111111111111111111111111111111111111111;
};

template<int M>
struct highp_float {
	highp_float() = default;

	highp_float(double value) {
		float64 tmp = value;
		sign = !!tmp.sign();
		exponent = (tmp.exponent() + exponent_offset) - float64::exponent_offset;
		mantissa.template as<uint64_t>(M / 2 - 1) = tmp.mantissa() << (62 - float64::mantissa_bits);
	}

	void operator=(double value) {
		float64 tmp = value;
		mantissa = {};
		sign = !!tmp.sign();
		exponent = (tmp.exponent() + exponent_offset) - float64::exponent_offset;
		mantissa.template as<uint64_t>(M / 2 - 1) = tmp.mantissa() << (62 - float64::mantissa_bits);
	}

	operator double() const {
		return (sign ? -1 : 1) *
			(double)std::pow(2ull, (int64_t)exponent - exponent_offset) *
			(1ull + (double)(mantissa.template as<uint64_t>(M / 2 - 1) >> (62 - float64::mantissa_bits)) / (1ull << float64::mantissa_bits));
	}

	const highp_float& operator+=(highp_float it) {
		if (exponent < it.exponent)
			std::swap(*this, it);

		it.mantissa.template set<mantissa_bits, 1>();
		uint64_t scale = exponent - it.exponent;
		it.mantissa >>= (int)scale;
		mantissa.template set<mantissa_bits, 1>();


		if (it.sign == sign)
			mantissa += it.mantissa;
		else {
			if (scale || compare(mantissa, it.mantissa) != -1) {
				mantissa = mantissa - it.mantissa;
			}
			else {
				sign = !sign;
				mantissa = it.mantissa - mantissa;
			}
		}

		if (mantissa.template get<mantissa_bits + 1>()) {
			exponent++;
			mantissa >>= 1;
		}

		int shift = mantissa.bit_scan_reverse();
		if (shift != -1) {
			exponent -= mantissa_bits - shift;
			mantissa <<= mantissa_bits - shift;
		}
		else {
			exponent = 0;
		}

		mantissa.template set<mantissa_bits, 0>();

		return *this;
	}

	const highp_float& operator-=(highp_float it) {
		it.sign = !it.sign;
		return *this += it;
	}

	const highp_float& operator*=(highp_float it) {
		sign = sign ^ it.sign;

		uint64_t exp = it.exponent;
		if (exp > exponent_offset)
			exponent += exp - exponent_offset;
		else
			exponent -= exponent_offset - exp;

		mantissa.template set<mantissa_bits, 1>();
		it.mantissa.template set<mantissa_bits, 1>();
		mantissa >>= mantissa_bits + 1 - Mbits / 2;
		it.mantissa >>= mantissa_bits + 1 - Mbits / 2;
		mantissa *= it.mantissa;

		if (mantissa.template get<mantissa_bits + 1>()) {
			exponent++;
			mantissa >>= 1;
		}

		mantissa.template set<mantissa_bits, 0>();

		return *this;
	}

	const highp_float& operator/=(highp_float it) {
		sign = sign ^ it.sign;

		uint64_t exp = it.exponent;
		if (exp > exponent_offset)
			exponent -= exp - exponent_offset;
		else
			exponent += exponent_offset - exp;

		uinteger<M> man;
		int bit = mantissa_bits;
		mantissa.template set<mantissa_bits, 1>();
		it.mantissa.template set<mantissa_bits, 1>();
		while (bit >= 0) {
			int cmp = compare(mantissa, it.mantissa);
			if (cmp != -1) {
				man.template set<1>(bit);
				if (cmp == 0)break;
				mantissa -= it.mantissa;
			}
			else {
				bit--;
				mantissa <<= 1;
			}
		}

		mantissa = man;

		int shift = mantissa_bits - mantissa.bit_scan_reverse();
		exponent -= shift;
		mantissa <<= shift;
		mantissa.template set<mantissa_bits, 0>();

		return *this;
	}

	friend highp_float operator+(highp_float lhs, const highp_float& rhs) {
		return lhs += rhs;
	}
	friend highp_float operator-(highp_float lhs, const highp_float& rhs) {
		return lhs -= rhs;
	}
	friend highp_float operator*(highp_float lhs, const highp_float& rhs) {
		return lhs *= rhs;
	}
	friend highp_float operator/(highp_float lhs, const highp_float& rhs) {
		return lhs /= rhs;
	}

	bool operator>(const int log2) const {
		if (sign)return false;

		if (exponent > log2 + exponent_offset)return true;

		return false;
	}

	friend highp_float sqr(highp_float fp) {
		fp.sign = 0;

		uint64_t exp = fp.exponent;
		if (exp > exponent_offset)
			fp.exponent += exp - exponent_offset;
		else
			fp.exponent -= exponent_offset - exp;

		fp.mantissa.template set<mantissa_bits, 1>();
		fp.mantissa >>= mantissa_bits + 1 - Mbits / 2;
		fp.mantissa *= fp.mantissa;

		if (fp.mantissa.template get<mantissa_bits + 1>()) {
			fp.exponent++;
			fp.mantissa >>= 1;
		}

		fp.mantissa.template set<mantissa_bits, 0>();

		return fp;
	}

	friend highp_float mul2(highp_float fp) {
		fp.exponent++;
		return fp;
	}

	static const int Mbits = M * 32;
	static const uint64_t sign_bits = 1;
	static const uint64_t exponent_bits = 63;
	static const uint64_t mantissa_bits = Mbits - 2;
	static const int64_t exponent_offset = (1ull << (exponent_bits - 1ull)) - 1ull;
	bool sign = 0;
	int64_t exponent = 0;
	uinteger<M> mantissa;
};

template<typename T>
class complex {
public:
	complex() = default;
	complex(const T& real, const T& imag) :real(real), imag(imag) {}
	template<typename U>
	complex(const Vector2<U>& rhs) : real((T)rhs.x), imag((T)rhs.y) {}
	template<typename U>
	complex(const complex<U>& rhs) : real((T)rhs.real), imag((T)rhs.imag) {}
	friend complex operator+(const complex& lhs, const complex& rhs) {
		return { lhs.real + rhs.real,lhs.imag + rhs.imag };
	}
	friend complex operator*(const complex& lhs, const complex& rhs) {
		return { lhs.real * rhs.real - lhs.imag * rhs.imag,
			lhs.real * rhs.imag + lhs.imag * rhs.real };
	}
	friend complex sqr(const complex& cpx) {
		return { sqr(cpx.real) - sqr(cpx.imag),
			mul2(cpx.real * cpx.imag) };
	}
	friend complex mul2(const complex& cpx) {
		return { ::mul2(cpx.real),::mul2(cpx.imag) };
	}
	friend T norm(const complex& cpx) {
		return { ::sqr(cpx.real) + ::sqr(cpx.imag) };
	}
	T real, imag;
};

template<typename HighpType, typename LowpType>
int calc_ref_point(Vector2<HighpType> co, int max_iterations, complex<LowpType>* Xs) {
	complex<HighpType> X0 = co, X = X0;

	int i = 0;
	for (; i < max_iterations; i++) {
		Xs[i] = X;
		X = sqr(X) + X0;
		if (norm(X) > 128)break;
	}

	return i;
}

template<typename LowpType>
float approx_point(Vector2<LowpType> dif_co, int max_iterations, const complex<LowpType>* Xs, int& iterations) {
	complex<LowpType> delta0 = dif_co, delta = delta0, Y = delta0;

	int i = 0;
	for (; i < max_iterations; i++) {
		delta = mul2(Xs[i] * delta) + sqr(delta) + delta0;
		Y = Xs[i] + delta;
		if (norm(Y) > 128)break;
	}

	iterations = i;

	if (i < max_iterations) {
		float nu = std::log(std::log2((float)norm(Y))) / std::log(2.0f);
		return (i + 1.0f - nu) / max_iterations;
	}
	return 1.0f;
}

int main() {
	print("hold  MOUSE LEFT/RIGHT  button to zoom  IN/OUT (mouse position is zooming center)");
	print("use  ARROW  to pan");
	print("press  SPACE  to save image");


	using HighpType = highp_float<32>;
	using LowpType = double;
	using xvec2 = Vector2<HighpType>;
	Image image{ {1280,720} };
	const int max_iterations{ 1024 * 32 };
	const int max_downres{ 8 };
	const int iteration_grow_speed = 64;
	const bool debug = false;
	/*
	using HighpType = highp_float<8>;
	using LowpType = float;
	using xvec2 = Vector2<HighpType>;
	Image image{ {1280,720} };
	const int max_iterations{ 1024 * 8 };
	const int max_downres{ 8 };
	const int iteration_grow_speed = 16;
	const bool debug = false;
	*/

	Viewer viewer{ image };
	int downres{ max_downres };
	bool reach_max_resolution{ false };
	HighpType scale{ 2.0 };
	xvec2 position{};
	HighpType scale_rate{ 2.0 };
	fvec2 dynamic_range{ 0.0f,1.0f };

	xvec2 ref_point{};
	ivec2 ref_coord{};
	xvec2 ref_point_candicate{};
	int candicate_max_iterations{};
	int last_ref_iterations{};
	bool any_candicate{};
	std::shared_ptr<complex<LowpType>[]> Xs{ new complex<LowpType>[max_iterations] };
	std::mutex mutex;

	calc_ref_point<HighpType, LowpType>(ref_point, max_iterations, Xs.get());

	int ref_skip_counter = 0;
	while (!viewer.should_close()) {
		float variance{}, mean{}, minmax[2]{ 1.0f,0.0f };
		int adaptive_iterations = std::min(512 + iteration_grow_speed * int(scale.exponent_offset - scale.exponent), max_iterations);

		bool recalculate_ref_point = any_candicate && (ref_skip_counter > 32 || last_ref_iterations != adaptive_iterations);
		if (recalculate_ref_point) {
			Timer calc_ref_timer;
			ref_point = ref_point_candicate;
			last_ref_iterations = calc_ref_point<HighpType, LowpType>(ref_point, adaptive_iterations, Xs.get());

			if (debug) {
				print("scale ", std::setw(12), scale, "  max iterations ", std::setw(6),
					adaptive_iterations, "  ref iterations ", std::setw(6), last_ref_iterations);
				print("reference point calculation time ", calc_ref_timer.duration(), "s\n");
			}

			ref_skip_counter = 0;
			candicate_max_iterations = 0;
			any_candicate = false;
		}
		ref_skip_counter++;


		if (!reach_max_resolution) {
			dvec2 dcenter;
			glfwGetCursorPos(viewer.window, &dcenter.x, &dcenter.y);
			ivec2 center{ (int)dcenter.x,image.size.y - 1 - (int)dcenter.y };

			parallel_for(image.size, [&](ivec2 p) {
				if (p.x % downres != 0 || p.y % downres != 0)return;

				xvec2 st = { (p.x - image.size.x / 2) / (double)image.size.y,
							 (p.y - image.size.y / 2) / (double)image.size.y };
				xvec2 pt = { position.x + st.x * scale,position.y + st.y * scale };

				int iterations = 0;

				float value = approx_point<LowpType>({ pt.x - ref_point.x, pt.y - ref_point.y }, adaptive_iterations, Xs.get(), iterations);
				{
					std::lock_guard<std::mutex> lk(mutex);
					if (iterations != 0 && iterations >= candicate_max_iterations) {
						if (iterations == candicate_max_iterations) {
							int dist0 = sqr(ref_coord.x - center.x) + sqr(ref_coord.y - center.y);
							int dist1 = sqr(p.x - center.x) + sqr(p.y - center.y);
							if (dist1 < dist0) {
								candicate_max_iterations = iterations;
								ref_point_candicate = pt;
								ref_coord = p;
								any_candicate = true;
							}
						}
						else {
							candicate_max_iterations = iterations;
							ref_point_candicate = pt;
							ref_coord = p;
							any_candicate = true;
						}
					}
				}

				if (std::isnan(value))value = 0;
				value = std::max(std::min(value, 1.0f), 0.0f);
				image[p].x = value;
				std::lock_guard<std::mutex> lk(mutex);
				mean += value;
				});




			mean /= image.size.x / downres * image.size.y / downres;
			parallel_for({ image.size.x / downres,image.size.y / downres }, [&](ivec2 p) {
				p.x *= downres; p.y *= downres;;
				std::lock_guard<std::mutex> lk(mutex);
				variance += std::abs(image[p].x - mean);
				});

			variance /= image.size.x / downres * image.size.y / downres;

			parallel_for(image.size, [&](ivec2 p) {
				if (p.x % downres != 0 || p.y % downres != 0)return;
				float dif = image[p].x - mean;
				float heat = std::exp(-std::abs(dif) / std::sqrt(variance));
				if (dif > 0)heat = 1.0f - heat / 2;
				else if (dif < 0)heat = heat / 2;
				else heat = 0.5f;

				image[p].x = heat;
				minmax[0] = std::min(minmax[0], heat);
				minmax[1] = std::max(minmax[1], heat);
				});


			float lerp_speed = (downres != 1) ? 0.04f : 1.0f;
			dynamic_range.x = lerp(lerp_speed, dynamic_range.x, minmax[0]);
			dynamic_range.y = lerp(lerp_speed, dynamic_range.y, minmax[1]);


			parallel_for(image.size, [&](ivec2 p) {
				if (p.x % downres != 0 || p.y % downres != 0)return;
				fvec3 cd = colormap((image[p].x - dynamic_range.x) / (dynamic_range.y - dynamic_range.x));
				image[p] = { cd.x * .8f,cd.y * .8f ,cd.z * .8f };
				});

			parallel_for(image.size, [&](ivec2 p) {
				if (p.x % downres != 0 || p.y % downres != 0)
					image[p] = image[{(p.x / downres)* downres, (p.y / downres)* downres}];
				});
		}




		if (debug && recalculate_ref_point)
			parallel_for(ivec2(41), [&](ivec2 p) {
			ivec2 co = { ref_coord.x + p.x - 20,ref_coord.y + p.y - 20 };
			if (co.x >= 0 && co.y >= 0 && co.x < image.size.x && co.y < image.size.y)
				image[co] = { 1, 0, 0 };
				});




		static int counter = 0;
		if (glfwGetKey(viewer.window, GLFW_KEY_SPACE))
			image.write_to_file("mandelbrot_image_" + std::to_string(counter++));


		viewer.render();


		if (max_downres != 1 && downres == 1)
			reach_max_resolution = true;
		else
			reach_max_resolution = false;
		if (downres >= 2)downres /= 2;




		static Timer timer;
		double px, py;
		glfwGetCursorPos(viewer.window, &px, &py);
		py = image.size.y - 1.0 - py;
		xvec2 st = { (px - image.size.x / 2) / (double)image.size.y,
					 (py - image.size.y / 2) / (double)image.size.y };

		if (glfwGetMouseButton(viewer.window, GLFW_MOUSE_BUTTON_1)) {
			HighpType scale_rate = 1.0 + 1.5 * timer.duration(0.1);
			position.x += st.x * (scale - scale / scale_rate);
			position.y += st.y * (scale - scale / scale_rate);
			scale /= scale_rate;
			downres = max_downres;
		}
		if (glfwGetMouseButton(viewer.window, GLFW_MOUSE_BUTTON_2)) {
			HighpType scale_rate = 1.0 + 1.5 * timer.duration(0.1);
			position.x += st.x * (scale - scale * scale_rate);
			position.y += st.y * (scale - scale * scale_rate);
			scale *= scale_rate;
			downres = max_downres;
		}
		if (glfwGetKey(viewer.window, GLFW_KEY_UP)) {
			position.y += (HighpType)timer.duration(0.1) * (HighpType)scale;
			downres = max_downres;
		}
		if (glfwGetKey(viewer.window, GLFW_KEY_DOWN)) {
			position.y -= (HighpType)timer.duration(0.1) * (HighpType)scale;
			downres = max_downres;
		}
		if (glfwGetKey(viewer.window, GLFW_KEY_LEFT)) {
			position.x -= (HighpType)timer.duration(0.1) * (HighpType)scale;
			downres = max_downres;
		}
		if (glfwGetKey(viewer.window, GLFW_KEY_RIGHT)) {
			position.x += (HighpType)timer.duration(0.1) * (HighpType)scale;
			downres = max_downres;
		}
		timer.reset();
	}

	return 0;
}
