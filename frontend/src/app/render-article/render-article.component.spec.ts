import { HttpClientModule } from '@angular/common/http';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { RenderArticleComponent } from './render-article.component';
import { RouterTestingModule } from "@angular/router/testing";

describe('RenderArticleComponent', () => {
  let component: RenderArticleComponent;
  let fixture: ComponentFixture<RenderArticleComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ RenderArticleComponent ],
      imports: [
        HttpClientModule,
        RouterTestingModule
      ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(RenderArticleComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
