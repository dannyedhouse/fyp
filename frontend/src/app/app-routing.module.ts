import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { RenderArticleComponent } from './render-article/render-article.component';

const routes: Routes = [
  { path: '', component: HomeComponent},
  { path: 'summary', component: RenderArticleComponent},
  { path: '**', component: HomeComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
