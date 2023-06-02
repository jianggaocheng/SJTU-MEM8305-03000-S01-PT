<template>
  <v-container fluid class="login-container">
    <v-row align="center" justify="center">
      <v-col cols="12" sm="10" md="6" lg="4">
        <v-card class="login-card">
          <div class="login-header">
            <img class="logo" src="@/assets/1_forest.png" alt="Logo">
            <h2 class="system-name">Forest Fire Alert System</h2>
          </div>
          <v-card-text>
            <v-form @submit.prevent="login">
              <v-text-field v-model="form.username" label="Username" outlined></v-text-field>
              <v-text-field v-model="form.password" label="Password" outlined type="password"></v-text-field>
              <v-btn type="submit" color="primary" block>Login</v-btn>
            </v-form>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  data() {
    return {
      form: {
        username: '',
        password: ''
      }
    };
  },
  methods: {
    login() {
      // 跳转到系统内部页面
      this.$router.push('/dashboard');
    }
  }
};
</script>

<style scoped>
.login-container {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
}

.login-card {
  padding: 24px;
}

.login-header {
  text-align: center;
  margin-bottom: 24px;
}

.logo {
  width: 120px;
  height: 120px;
  margin-bottom: 16px;
}

.system-name {
  font-size: 20px;
  font-weight: bold;
  color: #333;
  margin: 0;
}

@media screen and (max-width: 600px) {
  .login-card {
    padding: 16px;
  }
}
</style>
