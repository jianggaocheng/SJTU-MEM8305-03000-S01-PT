<template>
  <v-app>
    <v-navigation-drawer app v-model="drawer" :clipped="true" :mini-variant.sync="miniVariant" permanent>
      <v-list dense>
        <v-list-item v-for="item in menuItems" :key="item.title" link @click="navigateTo(item.route)">
          <v-list-item-icon>
            <v-icon>{{ item.icon }}</v-icon>
          </v-list-item-icon>
          <v-list-item-content>
            <v-list-item-title>{{ item.title }}</v-list-item-title>
          </v-list-item-content>
        </v-list-item>
      </v-list>
    </v-navigation-drawer>

    <v-app-bar app flat color="#ffffff">
      <v-app-bar-nav-icon @click.stop="drawer = !drawer"></v-app-bar-nav-icon>
      <div class="title-line"></div>
      <v-toolbar-title class="title">Forest Fire Alert System</v-toolbar-title>
    </v-app-bar>

    <v-main>
      <router-view />
    </v-main>
  </v-app>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';

const drawer = ref(false);
const miniVariant = ref(false);
const router = useRouter();

const menuItems = [
  { title: 'Dashboard', icon: 'mdi-view-dashboard', route: '/dashboard' },
  { title: 'Device List', icon: 'mdi-view-list', route: '/device-list' }
];

const navigateTo = (route) => {
  router.push(route);
};
</script>

<style>
.title {
  font-weight: 500;
}

.title-line {
  border-bottom: 1px solid #e0e0e0;
  margin-bottom: 8px;
}
</style>
