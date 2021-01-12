import { Component, OnInit, ViewChild } from '@angular/core';
import { ActivatedRoute, Router } from "@angular/router";
import { NgbAlert } from '@ng-bootstrap/ng-bootstrap';
import { ApiService } from '../api.service';

@Component({
  selector: 'app-render-article',
  templateUrl: './render-article.component.html',
  styleUrls: ['./render-article.component.scss']
})
export class RenderArticleComponent implements OnInit {
  @ViewChild('errorAlert', {static: false}) errorAlert: NgbAlert;

  constructor(private route: ActivatedRoute, private apiService: ApiService, private router: Router) { }
  
  url: string;
  summary: any;
  loading: boolean;
  errorMessage = "";

  ngOnInit(): void {
    this.url = this.route.snapshot.queryParamMap.get("url")
    this.route.queryParamMap.subscribe(queryParams => {
      this.url = queryParams.get("url")
    })
    this.loading = true;
    this.getSummary(this.url);
  }

  private getSummary(url: string): void {
    if (this.url == null) {
      this.displayAlert("No URL parameter specified");
    } else {
      this.apiService.getSummary(url).subscribe((value: any)=>{
        if (Object.keys(value).length == 0) {
          this.displayAlert("No results found. Please try a different URL");
        } else {
          this.summary = value;
          this.loading = false;
        }
      }, error => {
        console.log(error);
        this.displayAlert("API is not currently available. Please try again later.");
      });
    }
  }

  returnHome(): void {
    this.router.navigate(["/home"]);
  }

  displayAlert(error: string): void {
    this.errorMessage = error;
    setTimeout(() => {
      this.closeAlert();
    }, 3000);
  }

  closeAlert(): void {
    this.errorMessage = "";
    this.returnHome();
  }
}
